#!/usr/bin/env python3
"""Measure CAMPPlus speaker assignment against held-out episodes.

Feature path mirrors the one the model already uses for its speaker latent
(indextts/infer_v2_modded.py): 16 kHz mono, 80-bin kaldi fbank, dither off,
mean subtracted over time, then CAMPPlus -> 192 dims.

A speaker's centroid is the L2-normalised mean of their enrolment embeddings and
a trial is scored by cosine against every centroid. Two quantities matter and
they pull against each other:

  positives  clips from episodes withheld from enrolment; top-1 must be right
  impostors  clips from speakers absent from the gallery; must be rejected

Assigning a clip to the wrong speaker builds a training pair whose prompt is
somebody else, and a prompt that does not predict the target is a prompt the
model learns to ignore — the same failure that made v1 read emotion off the text.
Rejecting a clip only costs data, and there are ~390 h to spend. So the threshold
is picked for precision; the sweep reports what each choice costs in recall.

  SPLIT=split.json AUDIO_ROOT=... CAMPPLUS=... OUT=result.npz \
      python tools/speaker_verify.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus  # noqa: E402

SPLIT = os.environ["SPLIT"]
# Manifest paths are relative to the manifest directory, so that is the anchor.
AUDIO_ROOT = os.environ["AUDIO_ROOT"]
CAMPPLUS = os.environ["CAMPPLUS"]
OUT = os.environ.get("OUT", "speaker_verify.npz")
WORKERS = int(os.environ.get("WORKERS", "16"))
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


class Clips(Dataset):
    """Decode, resample and featurise on worker processes; the GPU work is tiny."""

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = os.path.normpath(os.path.join(AUDIO_ROOT, self.paths[i]))
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return i, None
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[1] < 1600:  # under 0.1 s carries no usable voiceprint
            return i, None
        feat = torchaudio.compliance.kaldi.fbank(
            wav, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        return i, feat - feat.mean(dim=0, keepdim=True)


def collate(batch):
    return batch


@torch.no_grad()
def embed(model, paths: list[str], tag: str) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(paths), 192), dtype=np.float32)
    ok = np.zeros(len(paths), dtype=bool)
    loader = DataLoader(
        Clips(paths), batch_size=32, num_workers=WORKERS, collate_fn=collate,
        prefetch_factor=4 if WORKERS else None,
    )
    done = 0
    t0 = time.time()
    for batch in loader:
        for i, feat in batch:
            done += 1
            if feat is None:
                continue
            out[i] = model(feat.to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
            ok[i] = True
        if done % 4000 < 32:
            print(f"    {tag} {done:,}/{len(paths):,}  {done / (time.time() - t0):.0f}/s",
                  flush=True)
    print(f"    {tag} done {int(ok.sum()):,}/{len(paths):,} in {time.time() - t0:.0f}s",
          flush=True)
    return out, ok


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def as_norm(scores: np.ndarray, cohort_mu: np.ndarray, cohort_sd: np.ndarray,
            topk: int = 50) -> np.ndarray:
    """Adaptive score normalisation.

    Raw cosine to a centroid mixes speaker identity with whatever else a clip
    happens to share with everything in the gallery — recording chain, noise
    floor, how much of it is actually voiced. The symptom is that in-gallery and
    out-of-gallery clips end up with the same score distribution, which is what
    the first run showed: at every threshold, impostors were accepted at almost
    exactly the rate positives were.

    Both sides get z-scored against a cohort. The trial side uses the clip's own
    scores against the gallery, which removes a per-clip offset; the enrolment
    side uses precomputed statistics of each centroid against other speakers'
    clips, which removes a per-speaker one.
    """
    top = np.sort(scores, axis=1)[:, -topk:]
    mu = top.mean(axis=1, keepdims=True)
    sd = np.maximum(top.std(axis=1, keepdims=True), 1e-6)
    return 0.5 * ((scores - mu) / sd + (scores - cohort_mu) / cohort_sd)


def main() -> None:
    split = json.load(open(SPLIT, encoding="utf-8"))
    gallery = split["gallery"]
    speakers = sorted(gallery)
    index = {s: i for i, s in enumerate(speakers)}

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(CAMPPLUS, map_location="cpu"))
    model = model.to(DEVICE).eval()
    print(f"  device {DEVICE}  workers {WORKERS}  gallery {len(speakers)}\n", flush=True)

    enrol_paths, enrol_spk = [], []
    pos_paths, pos_spk, pos_dur = [], [], []
    for s in speakers:
        for p in gallery[s]["enrol"]:
            enrol_paths.append(p)
            enrol_spk.append(index[s])
        for p, d in gallery[s]["positive"]:
            pos_paths.append(p)
            pos_spk.append(index[s])
            pos_dur.append(d)
    imp_paths, imp_dur = [], []
    for clips in split["impostors"].values():
        for p, d in clips:
            imp_paths.append(p)
            imp_dur.append(d)

    E, e_ok = embed(model, enrol_paths, "enrol")
    P, p_ok = embed(model, pos_paths, "positive")
    I, i_ok = embed(model, imp_paths, "impostor")

    enrol_spk = np.asarray(enrol_spk)
    pos_spk = np.asarray(pos_spk)
    centroids = np.zeros((len(speakers), 192), dtype=np.float32)
    enrolled = np.zeros(len(speakers), dtype=bool)
    for i in range(len(speakers)):
        m = e_ok & (enrol_spk == i)
        if m.any():
            centroids[i] = unit(E[m]).mean(0)
            enrolled[i] = True
    centroids = unit(centroids)

    pos_raw = unit(P[p_ok]) @ centroids.T
    imp_raw = unit(I[i_ok]) @ centroids.T
    truth = pos_spk[p_ok]

    # Per-centroid statistics against other speakers' enrolment clips, so each
    # centroid's own scale comes out of the score.
    En = unit(E[e_ok])
    en_spk = enrol_spk[e_ok]
    cs = En @ centroids.T
    cohort_mu = np.zeros((1, len(speakers)), dtype=np.float32)
    cohort_sd = np.ones((1, len(speakers)), dtype=np.float32)
    for i in range(len(speakers)):
        other = np.sort(cs[en_spk != i, i])[-200:]
        if other.size >= 20:
            cohort_mu[0, i] = other.mean()
            cohort_sd[0, i] = max(float(other.std()), 1e-6)

    if os.environ.get("SCORE_NORM", "1") == "1":
        pos_scores = as_norm(pos_raw, cohort_mu, cohort_sd)
        imp_scores = as_norm(imp_raw, cohort_mu, cohort_sd)
        thresholds = np.arange(0.0, 8.1, 0.5)
        print("\n  scoring: AS-norm (raw cosine also reported below)")
    else:
        pos_scores, imp_scores = pos_raw, imp_raw
        thresholds = np.arange(0.30, 0.91, 0.05)

    pos_top, pos_best = pos_scores.max(1), pos_scores.argmax(1)
    imp_top = imp_scores.max(1)
    raw_top1 = float((pos_raw.argmax(1) == truth).mean())
    pdur = np.asarray(pos_dur)[p_ok]
    idur = np.asarray(imp_dur)[i_ok]

    print(f"\n  enrolled {int(enrolled.sum())}/{len(speakers)} speakers | "
          f"positives {int(p_ok.sum()):,} | impostors {int(i_ok.sum()):,}")

    # Duration is the variable that decides whether this can work at all: tai8's
    # clips are far shorter than the resegment clips the labels are needed for,
    # so a pooled number would answer the wrong question.
    buckets = [(0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 99.0)]
    print(f"\n  {'trial dur':>11}{'n':>7}{'top-1':>9}   sweep (thresh: correct / impostor FA)")
    print("  " + "-" * 74)
    for lo, hi in buckets:
        pm, im = (pdur >= lo) & (pdur < hi), (idur >= lo) & (idur < hi)
        if pm.sum() < 30:
            continue
        top1 = float((pos_best[pm] == truth[pm]).mean())
        cells = []
        for th in (thresholds[len(thresholds) // 2], thresholds[2 * len(thresholds) // 3],
                   thresholds[5 * len(thresholds) // 6]):
            acc = pm & (pos_top >= th)
            prec = float((pos_best[acc] == truth[acc]).mean()) if acc.sum() >= 20 else float("nan")
            far = float((imp_top[im] >= th).mean()) if im.sum() >= 20 else float("nan")
            cells.append(f"{th:.2f}: {100 * prec:4.0f}% / {100 * far:4.0f}%")
        label = f"{lo:.1f}-{hi:.1f}s" if hi < 90 else f">={lo:.1f}s"
        print(f"  {label:>11}{int(pm.sum()):>7}{100 * top1:>8.1f}%   {'  '.join(cells)}")

    print(f"\n  pooled top-1  {100 * (pos_best == truth).mean():.1f}%"
          f"   (raw cosine {100 * raw_top1:.1f}%)\n")
    print(f"  {'thresh':>7}{'accepted':>10}{'correct':>10}{'impostor FA':>13}{'yield':>9}")
    print("  " + "-" * 49)
    best = None
    for th in thresholds:
        accepted = pos_top >= th
        if not accepted.any():
            continue
        precision = float((pos_best[accepted] == truth[accepted]).mean())
        far = float((imp_top >= th).mean())
        recall = float(accepted.mean())
        print(f"  {th:>7.2f}{100 * recall:>9.1f}%{100 * precision:>9.1f}%"
              f"{100 * far:>12.1f}%{100 * recall * precision:>8.1f}%")
        if precision >= 0.98 and far <= 0.02 and (best is None or recall > best[1]):
            best = (float(th), recall, precision, far)

    if best:
        th, recall, precision, far = best
        print(f"\n  best point with precision >=98% and impostor FA <=2%:")
        print(f"    threshold {th:.2f}  accepts {100 * recall:.1f}%  "
              f"correct {100 * precision:.1f}%  impostor FA {100 * far:.1f}%")
    else:
        print("\n  no threshold reached precision >=98% with impostor FA <=2%")

    np.savez(
        OUT, centroids=centroids, speakers=np.asarray(speakers), enrolled=enrolled,
        pos_top=pos_top, pos_best=pos_best, pos_truth=truth, imp_top=imp_top,
        pos_dur=pdur, imp_dur=idur,
    )
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
