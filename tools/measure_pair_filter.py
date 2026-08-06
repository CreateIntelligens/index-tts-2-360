#!/usr/bin/env python3
"""Would rejecting low-similarity pairs remove the ones with the wrong prompt?

Long-form output from the finetuned 1.5 shifts voice at every sentence boundary
the splitter creates, while the stock Chinese 1.5 does not — same code, same
splits, same seed, same reference clip, only the weights differ. Boundaries are
inherent: each sentence is sampled independently with no acoustic context. What
decides whether they are audible is how tightly the model ties conditioning to
timbre, and finetuning loosened that. The likely reason is that tai8's speaker
labels are automatic — the workers corrected transcripts and timings, not
speakers — so a share of pairs hand the model a prompt spoken by somebody else,
which teaches exactly that looseness.

Identifying who is speaking failed earlier: 63.7% top-1 over 147 identities.
Deciding whether *these two clips* are the same person is a much smaller
question, and it is the only one pairing needs. This measures whether a cosine
threshold on CAMPPlus embeddings separates the good pairs from the bad, and what
it costs in data.

There is no human speaker ground truth to check against. The stand-in is
agreement with the pyannote labels a colleague produced independently, matched
to tai8 by transcript within an episode. Two acoustic methods agreeing is weaker
evidence than a human would be, and correlated errors would flatter the result;
it is reported as agreement rather than accuracy for that reason.

Pairs spanning episodes are left out of the check: pyannote's cross-episode
linking measured 18.13% pairwise precision, so it cannot adjudicate those.

  python tools/measure_pair_filter.py
"""

from __future__ import annotations

import collections
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus  # noqa: E402

SP = os.environ.get(
    "SCRATCH",
    "/tmp/claude-1000/-home-hank-index-tts-2-360/de1d5f9f-02c2-4f53-b990-c4d1bdda0f21/scratchpad",
)
TAI8 = os.environ.get("TAI8_MANIFESTS", "/mnt/nas/dataset202607_1/tai8/manifests/index_tts")
AUTO_ROOT = os.environ.get("AUTO_ROOT", "/mnt/nas/ml-material/segment_data/output/drama1")
AUTO_INDEX = os.environ.get("AUTO_INDEX", f"{SP}/seg_drama1.txt")
CAMPPLUS = os.environ["CAMPPLUS"]
OUT = os.environ.get("OUT", f"{SP}/pair_filter.npz")
PAIRS = int(os.environ.get("PAIRS", "15000"))
MIN_DUR = float(os.environ.get("MIN_DUR", "1.0"))
MAX_DUR = float(os.environ.get("MAX_DUR", "15.0"))
WORKERS = int(os.environ.get("WORKERS", "12"))
SEED = int(os.environ.get("SEED", "1234"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PUNCT = re.compile(r"[\s，。！？、,.!?~\-—…「」『』:：;；()（）]")
EPISODE_RE = re.compile(r"segments/drama\d/([^/]+)/([^/]+)/")


def norm(t: str) -> str:
    return PUNCT.sub("", t)


def read_auto_episode(args):
    ep, paths = args
    out = []
    for rel in paths:
        t = os.path.join(AUTO_ROOT, rel[:-4] + ".normalized.txt")
        try:
            txt = open(t, encoding="utf-8").read().strip()
        except OSError:
            continue
        if txt:
            out.append((rel.split("/")[0], norm(txt)))
    return ep, out


class Clips(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            wav, sr = torchaudio.load(self.paths[i])
        except Exception:
            return i, None
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[1] < 1600:
            return i, None
        feat = torchaudio.compliance.kaldi.fbank(
            wav, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        return i, feat - feat.mean(dim=0, keepdim=True)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main() -> None:
    rng = random.Random(SEED)

    # tai8 side: clip -> speaker, episode, path, normalised text
    clips = []
    for split in ("train", "val"):
        path = f"{TAI8}/{split}_manifest.jsonl"
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            if "drama1" not in d["audio"] or not MIN_DUR <= d["duration"] <= MAX_DUR:
                continue
            m = EPISODE_RE.search(d["audio"])
            if not m:
                continue
            clips.append((d["speaker"], m.group(2),
                          os.path.normpath(os.path.join(TAI8, d["audio"])), norm(d["text"])))
    print(f"  tai8 drama1 clips: {len(clips):,}", flush=True)

    labelled = {ep for _, ep, _, _ in clips}
    by_ep_auto = collections.defaultdict(list)
    for line in open(AUTO_INDEX, encoding="utf-8"):
        rel = line.strip()
        p = rel.split("/")
        if len(p) == 3 and p[1] in labelled:
            by_ep_auto[p[1]].append(rel)

    print("  reading the auto-labelled transcripts ...", flush=True)
    auto_text_to_spk = {}
    with ProcessPoolExecutor(16) as ex:
        for ep, rows in ex.map(read_auto_episode, by_ep_auto.items(), chunksize=2):
            counts = collections.Counter(t for _, t in rows)
            for spk, t in rows:
                # A line occurring twice in an episode cannot anchor anything.
                if counts[t] == 1 and len(t) >= 4:
                    auto_text_to_spk[(ep, t)] = spk

    # Only clips both corpora agree on the identity of can adjudicate a pair.
    resolved = []
    for spk, ep, path, text in clips:
        auto = auto_text_to_spk.get((ep, text))
        if auto:
            resolved.append((spk, ep, path, auto))
    print(f"  clips with a pyannote opinion: {len(resolved):,}", flush=True)

    # Pairs the way training builds them: same tai8 speaker, same episode so the
    # second opinion applies.
    buckets = collections.defaultdict(list)
    for spk, ep, path, auto in resolved:
        buckets[(spk, ep)].append((path, auto))
    usable = [v for v in buckets.values() if len(v) >= 2]
    print(f"  speaker-episode buckets with >=2 clips: {len(usable):,}", flush=True)

    pairs = []
    while len(pairs) < PAIRS and usable:
        b = rng.choice(usable)
        a, c = rng.sample(b, 2)
        pairs.append((a[0], c[0], a[1] == c[1]))
    print(f"  sampled {len(pairs):,} pairs; "
          f"{100 * sum(p[2] for p in pairs) / len(pairs):.1f}% have pyannote agreement\n",
          flush=True)

    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    index = {p: i for i, p in enumerate(paths)}
    print(f"  embedding {len(paths):,} clips ...", flush=True)

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(CAMPPLUS, map_location="cpu"))
    model = model.to(DEVICE).eval()

    emb = np.zeros((len(paths), 192), dtype=np.float32)
    ok = np.zeros(len(paths), dtype=bool)
    loader = DataLoader(Clips(paths), batch_size=32, num_workers=WORKERS,
                        collate_fn=lambda b: b, prefetch_factor=4 if WORKERS else None)
    done, t0 = 0, time.time()
    with torch.no_grad():
        for batch in loader:
            for i, feat in batch:
                done += 1
                if feat is None:
                    continue
                emb[i] = model(feat.to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
                ok[i] = True
            if done % 5000 < 32:
                print(f"    {done:,}/{len(paths):,}  {done / (time.time() - t0):.0f}/s", flush=True)
    emb = unit(emb)

    sims, agree = [], []
    for a, b, same in pairs:
        ia, ib = index[a], index[b]
        if ok[ia] and ok[ib]:
            sims.append(float(emb[ia] @ emb[ib]))
            agree.append(same)
    sims = np.asarray(sims)
    agree = np.asarray(agree)
    print(f"\n  scored {len(sims):,} pairs")
    print(f"  cosine: p10 {np.percentile(sims, 10):.3f}  median {np.median(sims):.3f}  "
          f"p90 {np.percentile(sims, 90):.3f}")
    print(f"  agreement among all pairs: {100 * agree.mean():.1f}%\n")

    print(f"  {'threshold':>10}{'kept':>9}{'agreement':>12}{'pairs kept':>13}")
    print("  " + "-" * 45)
    base = agree.mean()
    for th in np.arange(0.0, 0.85, 0.05):
        keep = sims >= th
        if keep.sum() < 50:
            continue
        print(f"  {th:>10.2f}{100 * keep.mean():>8.1f}%{100 * agree[keep].mean():>11.1f}%"
              f"{int(keep.sum()):>13,}")

    np.savez(OUT, sims=sims, agree=agree)
    print(f"\n  wrote {OUT}")
    print(f"  baseline agreement without filtering: {100 * base:.1f}%")


if __name__ == "__main__":
    main()
