#!/usr/bin/env python3
"""Can speakers be separated *within one episode*, well enough to build pairs?

Matching resegment clips to tai8's 147 recurring identities failed: top-1 was 63%
and holding precision at 98% left only 3% of clips. That is a cross-session,
open-set problem over many identities.

Clustering inside a single episode is a different and much smaller problem — a
handful of characters, one recording session — and it is all that pairing
actually needs. A pair only has to satisfy "these two clips are the same person";
it does not need to know *which* person, and it does not need to agree with
tai8's show-level numbering.

So the metric here is the one that decides whether a pair is sound:

  pairwise precision  of the clip pairs this clustering would allow,
                      the fraction that really are the same speaker
  coverage            fraction of clips landing in a cluster of size >= 2,
                      since a singleton yields no pair at all

Ground truth is tai8's own speaker labels on episodes the workers labelled.

  SPLIT-free; reads the tai8 manifests directly.
  EPISODES=40 MAX_CLIPS=400 python tools/speaker_cluster_eval.py
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

import numpy as np
import torch
import torchaudio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus  # noqa: E402

MANIFESTS = os.environ.get(
    "TAI8_MANIFESTS", "/mnt/shared/p06/dataset202607_1/tai8/manifests/index_tts"
)
CAMPPLUS = os.environ["CAMPPLUS"]
OUT = os.environ.get("OUT", "cluster_eval.npz")
EPISODES = int(os.environ.get("EPISODES", "40"))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", "400"))
MIN_SPEAKERS = int(os.environ.get("MIN_SPEAKERS", "3"))
MIN_CLIPS = int(os.environ.get("MIN_CLIPS", "60"))
WORKERS = int(os.environ.get("WORKERS", "32"))
SEED = int(os.environ.get("SEED", "1234"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPISODE_RE = re.compile(r"segments/(drama\d)/([^/]+)/([^/]+)/")


class Clips(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = os.path.normpath(os.path.join(MANIFESTS, self.paths[i]))
        try:
            wav, sr = torchaudio.load(path)
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


@torch.no_grad()
def embed(model, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(paths), 192), dtype=np.float32)
    ok = np.zeros(len(paths), dtype=bool)
    loader = DataLoader(Clips(paths), batch_size=32, num_workers=WORKERS,
                        collate_fn=lambda b: b, prefetch_factor=4 if WORKERS else None)
    for batch in loader:
        for i, feat in batch:
            if feat is None:
                continue
            out[i] = model(feat.to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
            ok[i] = True
    return out, ok


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main() -> None:
    rows = []
    for split in ("train", "val"):
        path = f"{MANIFESTS}/{split}_manifest.jsonl"
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            m = EPISODE_RE.search(d["audio"])
            if m and 1.0 <= d["duration"] <= 15.0:
                rows.append((m.group(1), m.group(3), d["speaker"], d["audio"]))

    by_ep: dict[tuple, list] = collections.defaultdict(list)
    for drama, ep, spk, audio in rows:
        by_ep[(drama, ep)].append((spk, audio))

    usable = [
        (k, v) for k, v in by_ep.items()
        if len(v) >= MIN_CLIPS and len({s for s, _ in v}) >= MIN_SPEAKERS
    ]
    usable.sort(key=lambda kv: -len(kv[1]))
    rng = np.random.default_rng(SEED)
    chosen = usable[:EPISODES]
    print(f"  episodes with >={MIN_CLIPS} clips and >={MIN_SPEAKERS} speakers: {len(usable)}")
    print(f"  evaluating {len(chosen)}\n", flush=True)

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(CAMPPLUS, map_location="cpu"))
    model = model.to(DEVICE).eval()

    thresholds = np.arange(0.20, 0.85, 0.05)
    same_pairs = np.zeros(len(thresholds))
    true_pairs = np.zeros(len(thresholds))
    covered = np.zeros(len(thresholds))
    total_clips = 0
    per_ep = []

    for n, ((drama, ep), items) in enumerate(chosen, 1):
        if len(items) > MAX_CLIPS:
            idx = rng.choice(len(items), MAX_CLIPS, replace=False)
            items = [items[i] for i in idx]
        labels = np.array([s for s, _ in items])
        X, ok = embed(model, [a for _, a in items])
        if ok.sum() < MIN_CLIPS:
            continue
        X, labels = unit(X[ok]), labels[ok]
        total_clips += len(labels)

        d = 1.0 - X @ X.T
        np.fill_diagonal(d, 0.0)
        d = np.clip((d + d.T) / 2, 0, None)
        Z = linkage(squareform(d, checks=False), method="average")

        ep_row = []
        for ti, th in enumerate(thresholds):
            pred = fcluster(Z, t=th, criterion="distance")
            sp = tp = cov = 0
            for c in np.unique(pred):
                m = pred == c
                k = int(m.sum())
                if k < 2:
                    continue
                cov += k
                sp += k * (k - 1) // 2
                counts = collections.Counter(labels[m])
                tp += sum(v * (v - 1) // 2 for v in counts.values())
            same_pairs[ti] += sp
            true_pairs[ti] += tp
            covered[ti] += cov
            ep_row.append((sp, tp, cov))
        per_ep.append(((drama, ep), len(labels), len(set(labels)), ep_row))
        if n % 5 == 0:
            print(f"    {n}/{len(chosen)} episodes, {total_clips:,} clips", flush=True)

    print(f"\n  {total_clips:,} clips over {len(per_ep)} episodes")
    print(f"  mean speakers per episode "
          f"{np.mean([s for _, _, s, _ in per_ep]):.1f}\n")
    print(f"  {'dist':>6}{'pair precision':>16}{'coverage':>11}{'usable pairs':>14}")
    print("  " + "-" * 47)
    best = None
    for ti, th in enumerate(thresholds):
        if same_pairs[ti] == 0:
            continue
        prec = true_pairs[ti] / same_pairs[ti]
        cov = covered[ti] / max(total_clips, 1)
        print(f"  {th:>6.2f}{100 * prec:>15.1f}%{100 * cov:>10.1f}%"
              f"{int(true_pairs[ti]):>14,}")
        if prec >= 0.98 and (best is None or true_pairs[ti] > best[3]):
            best = (float(th), prec, cov, true_pairs[ti])

    if best:
        th, prec, cov, tp = best
        print(f"\n  best point with pair precision >=98%:")
        print(f"    distance {th:.2f}  precision {100 * prec:.1f}%  "
              f"coverage {100 * cov:.1f}%  correct pairs {int(tp):,}")
    else:
        print("\n  no distance threshold reached 98% pair precision")

    np.savez(OUT, thresholds=thresholds, same_pairs=same_pairs,
             true_pairs=true_pairs, covered=covered, total_clips=total_clips)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
