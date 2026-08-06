#!/usr/bin/env python3
"""Give the resegment corpus speaker labels by clustering inside each episode.

That corpus has ~390 h over 1,092 episodes and no speaker labels, so it cannot
be paired: a prompt drawn from a different person does not predict the target,
and conditioning that does not predict the target is conditioning the model
learns to ignore.

Labels are produced per episode rather than per show. Pairing only needs "these
two clips are the same person" to be true; it never needs to know *which*
person, and it does not need to agree with tai8's numbering. Keeping clusters
inside an episode also makes the problem far smaller — a handful of characters
sharing one recording session, instead of open-set matching across hundreds of
identities, which measured 63.7% top-1 and was not usable.

Splitting one speaker across episodes is the safe failure: every pair drawn from
such a cluster is still correct, there are just fewer of them. Merging two
speakers is the dangerous one, so the linkage threshold is set for cluster
purity and singletons are dropped — they yield no pair anyway.

Episodes tai8 already covers are skipped: those were cut from the same source
audio and their transcripts were corrected by hand, so the corrected version is
the one to train on and counting both would duplicate the audio.

  CORPUS_ROOT=... OUT=manifest.jsonl python tools/assign_speakers_by_episode.py
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
import wave

import numpy as np
import torch
import torchaudio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus  # noqa: E402

CORPUS_ROOT = os.environ.get("CORPUS_ROOT", "/mnt/shared/p06/taigi_resegment/segments")
TAI8_MANIFESTS = os.environ.get(
    "TAI8_MANIFESTS", "/mnt/shared/p06/dataset202607_1/tai8/manifests/index_tts"
)
CAMPPLUS = os.environ["CAMPPLUS"]
OUT = os.environ.get("OUT", "resegment_manifest.jsonl")
STATS = os.environ.get("STATS", "")
# 0.50 sat at 84% pair precision with 80% of clips retained when this clustering
# was scored against tai8's labels; 0.40 buys 87% but throws away half the data.
DISTANCE = float(os.environ.get("DISTANCE", "0.50"))
MIN_DUR = float(os.environ.get("MIN_DUR", "1.0"))
MAX_DUR = float(os.environ.get("MAX_DUR", "15.0"))
MIN_CLUSTER = int(os.environ.get("MIN_CLUSTER", "2"))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", "4000"))
WORKERS = int(os.environ.get("WORKERS", "16"))
SHARD = int(os.environ.get("SHARD", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SKIP_TAI8_EPISODES = os.environ.get("SKIP_TAI8_EPISODES", "1") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPISODE_RE = re.compile(r"segments/(drama\d)/[^/]+/([^/]+)/")


class Clips(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            wav, sr = torchaudio.load(self.paths[i])
        except Exception:
            return i, None, 0.0
        dur = wav.shape[1] / sr
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[1] < 1600:
            return i, None, dur
        feat = torchaudio.compliance.kaldi.fbank(
            wav, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        return i, feat - feat.mean(dim=0, keepdim=True), dur


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def tai8_episodes() -> set[tuple[str, str]]:
    """(drama, episode) pairs the workers already transcribed and corrected."""
    covered = set()
    for split in ("train", "val"):
        path = f"{TAI8_MANIFESTS}/{split}_manifest.jsonl"
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            m = EPISODE_RE.search(json.loads(line)["audio"])
            if m:
                covered.add((m.group(1), m.group(2).lstrip("0") or "0"))
    return covered


def episodes() -> list[tuple[str, str, str]]:
    out = []
    for drama in sorted(os.listdir(CORPUS_ROOT)):
        d = os.path.join(CORPUS_ROOT, drama)
        if not os.path.isdir(d):
            continue
        for ep in sorted(os.listdir(d)):
            p = os.path.join(d, ep)
            if os.path.isdir(p):
                out.append((drama, ep, p))
    return out


@torch.no_grad()
def main() -> None:
    skip = tai8_episodes() if SKIP_TAI8_EPISODES else set()
    todo = []
    for drama, ep, path in episodes():
        if (drama, ep.lstrip("0") or "0") in skip:
            continue
        todo.append((drama, ep, path))
    todo = todo[SHARD::NUM_SHARDS]
    print(f"  tai8 covers {len(skip)} episodes; {len(todo)} to label "
          f"(shard {SHARD}/{NUM_SHARDS})", flush=True)

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(CAMPPLUS, map_location="cpu"))
    model = model.to(DEVICE).eval()

    handle = open(OUT, "w", encoding="utf-8")
    n_written = n_clips = n_singleton = 0
    clusters_per_ep = []
    t0 = time.time()

    for idx, (drama, ep, path) in enumerate(todo, 1):
        rows = []
        for name in sorted(os.listdir(path)):
            if not name.endswith(".wav"):
                continue
            txt = os.path.join(path, name[:-4] + ".normalized.txt")
            try:
                text = open(txt, encoding="utf-8").read().strip()
            except OSError:
                continue
            if text:
                rows.append((os.path.join(path, name), text))
        if len(rows) < MIN_CLUSTER or len(rows) > MAX_CLIPS:
            continue

        loader = DataLoader(Clips([p for p, _ in rows]), batch_size=32,
                            num_workers=WORKERS, collate_fn=lambda b: b,
                            prefetch_factor=4 if WORKERS else None)
        emb = np.zeros((len(rows), 192), dtype=np.float32)
        dur = np.zeros(len(rows), dtype=np.float32)
        ok = np.zeros(len(rows), dtype=bool)
        for batch in loader:
            for i, feat, d in batch:
                dur[i] = d
                if feat is None or not MIN_DUR <= d <= MAX_DUR:
                    continue
                emb[i] = model(feat.to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
                ok[i] = True
        if ok.sum() < MIN_CLUSTER:
            continue

        keep = np.flatnonzero(ok)
        X = unit(emb[keep])
        d = np.clip(1.0 - X @ X.T, 0, None)
        np.fill_diagonal(d, 0.0)
        Z = linkage(squareform((d + d.T) / 2, checks=False), method="average")
        labels = fcluster(Z, t=DISTANCE, criterion="distance")

        sizes = collections.Counter(labels)
        clusters_per_ep.append(len([c for c, n in sizes.items() if n >= MIN_CLUSTER]))
        for pos, c in zip(keep, labels):
            # A singleton cannot supply a prompt for itself without leaking the
            # target, so it would never form a pair.
            if sizes[c] < MIN_CLUSTER:
                n_singleton += 1
                continue
            audio, text = rows[pos]
            stem = os.path.splitext(os.path.basename(audio))[0]
            handle.write(json.dumps({
                "audio": audio,
                "duration": round(float(dur[pos]), 4),
                "id": f"{drama}_{ep}_{stem}",
                "language": "zh",
                "speaker": f"re_{drama}_{ep}_c{int(c):03d}",
                "text": text,
            }, ensure_ascii=False) + "\n")
            n_written += 1
        n_clips += int(ok.sum())

        if idx % 25 == 0:
            rate = idx / (time.time() - t0)
            print(f"    {idx}/{len(todo)} episodes  {n_written:,} kept  "
                  f"{rate * 60:.1f} ep/min", flush=True)

    handle.close()
    kept = 100 * n_written / max(n_clips, 1)
    print(f"\n  episodes labelled {len(clusters_per_ep)}")
    print(f"  clips embedded    {n_clips:,}")
    print(f"  kept              {n_written:,} ({kept:.1f}%)")
    print(f"  dropped singleton {n_singleton:,}")
    if clusters_per_ep:
        cp = sorted(clusters_per_ep)
        print(f"  clusters/episode  median {cp[len(cp) // 2]}  "
              f"min {cp[0]}  max {cp[-1]}")
    print(f"  wrote {OUT}")

    if STATS:
        json.dump({
            "episodes": len(clusters_per_ep), "clips": n_clips,
            "kept": n_written, "singletons": n_singleton,
            "distance": DISTANCE, "clusters_per_episode": clusters_per_ep,
        }, open(STATS, "w"))


if __name__ == "__main__":
    main()
