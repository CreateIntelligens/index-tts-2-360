#!/usr/bin/env python3
"""Lay out a held-out-episode trial for CAMPPlus speaker enrolment.

The resegment corpus (1,092 episodes, ~390 h) carries no speaker labels, and its
segment boundaries do not correspond to tai8's — the workers re-cut and re-typed
those, so nothing can be matched by timestamp or by text. What can be matched is
the voice: CAMPPlus maps a clip of any length to a 192-dim vector meant to be
invariant to content and duration.

Before trusting that on unlabelled episodes, measure it where the answer is
known. Enrol each speaker on some of their episodes and test on episodes held
out of enrolment, so a correct answer cannot come from having seen that
recording session.

Speakers appearing in a single episode cannot be tested that way. They are,
however, exactly who turns up in an unlabelled episode without being in the
gallery, so they serve as impostors: a usable threshold has to reject them.

  TAI8_MANIFESTS=... OUT=split.json python tools/build_speaker_split.py
"""

from __future__ import annotations

import collections
import json
import os
import random
import re
import sys

MANIFESTS = os.environ.get(
    "TAI8_MANIFESTS", "/mnt/shared/p06/dataset202607_1/tai8/manifests/index_tts"
)
OUT = os.environ.get("OUT") or (sys.argv[1] if len(sys.argv) > 1 else "speaker_split.json")

# A gallery speaker needs enough episodes that holding some out still leaves an
# enrolment set, and enough clips that the centroid is not one bad take.
MIN_EPISODES = int(os.environ.get("MIN_EPISODES", "3"))
MIN_UTTS = int(os.environ.get("MIN_UTTS", "30"))
# Centroids saturate well before a speaker's full set; capping keeps one large
# role from dominating the embedding cost.
MAX_ENROL = int(os.environ.get("MAX_ENROL", "150"))
MAX_POS = int(os.environ.get("MAX_POS", "40"))
HOLDOUT_FRAC = float(os.environ.get("HOLDOUT_FRAC", "0.25"))
IMPOSTOR_MIN_UTTS = int(os.environ.get("IMPOSTOR_MIN_UTTS", "5"))
IMPOSTOR_SPEAKERS = int(os.environ.get("IMPOSTOR_SPEAKERS", "400"))
IMPOSTOR_PER_SPK = int(os.environ.get("IMPOSTOR_PER_SPK", "5"))
MIN_DUR = float(os.environ.get("MIN_DUR", "1.0"))
MAX_DUR = float(os.environ.get("MAX_DUR", "15.0"))
SEED = int(os.environ.get("SEED", "1234"))

# tai8 clips are short — median 1.59 s, only 5% reach 3 s — because the workers
# cut on subtitle cues. Speaker embeddings degrade sharply below a couple of
# seconds, so trial clips carry their duration and the scoring reports accuracy
# per duration bucket. The resegment clips this has to generalise to average
# 3.1-3.9 s, i.e. roughly double, so a single pooled number over tai8's short
# clips would understate what is achievable there.

EPISODE_RE = re.compile(r"segments/(drama\d)/([^/]+)/([^/]+)/")


def read_rows() -> list[tuple[str, str, str, float]]:
    rows = []
    for split in ("train", "val"):
        path = f"{MANIFESTS}/{split}_manifest.jsonl"
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                d = json.loads(line)
                m = EPISODE_RE.search(d["audio"])
                if not m or not MIN_DUR <= d["duration"] <= MAX_DUR:
                    continue
                rows.append((d["speaker"], m.group(3), d["audio"], d["duration"]))
    return rows


def main() -> None:
    rows = read_rows()
    if not rows:
        raise SystemExit(f"no usable rows under {MANIFESTS}")

    by_spk: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for spk, ep, audio, dur in rows:
        by_spk[spk][ep].append((audio, dur))

    rng = random.Random(SEED)
    gallery: dict[str, dict] = {}
    impostors: list[tuple[str, list[str]]] = []
    for spk, eps in sorted(by_spk.items()):
        total = sum(len(v) for v in eps.values())
        if len(eps) >= MIN_EPISODES and total >= MIN_UTTS:
            ep_ids = sorted(eps)
            rng.shuffle(ep_ids)
            k = max(1, round(len(ep_ids) * HOLDOUT_FRAC))
            held, enrol_eps = ep_ids[:k], ep_ids[k:]
            enrol = [a for e in enrol_eps for a in eps[e]]
            positive = [a for e in held for a in eps[e]]
            if not enrol or not positive:
                continue
            rng.shuffle(enrol)
            rng.shuffle(positive)
            # Longest first, so the capped sample keeps the clips that can
            # actually carry a voiceprint rather than a random short draw.
            positive.sort(key=lambda x: -x[1])
            gallery[spk] = {
                "enrol": [a for a, _ in enrol[:MAX_ENROL]],
                "positive": [[a, round(d, 3)] for a, d in positive[:MAX_POS]],
                "enrol_episodes": sorted(enrol_eps),
                "held_out_episodes": sorted(held),
                "total_utts": total,
            }
        elif len(eps) == 1 and total >= IMPOSTOR_MIN_UTTS:
            impostors.append((spk, sorted(next(iter(eps.values())), key=lambda x: -x[1])))

    rng.shuffle(impostors)
    imp = {
        spk: [[a, round(d, 3)] for a, d in clips[:IMPOSTOR_PER_SPK]]
        for spk, clips in impostors[:IMPOSTOR_SPEAKERS]
    }

    n_enrol = sum(len(v["enrol"]) for v in gallery.values())
    n_pos = sum(len(v["positive"]) for v in gallery.values())
    n_imp = sum(len(v) for v in imp.values())
    covered = sum(v["total_utts"] for v in gallery.values())

    print(f"  manifests                 {MANIFESTS}")
    print(f"  speakers ({MIN_DUR}-{MAX_DUR}s clips)   {len(by_spk):,}")
    print(f"  gallery (>={MIN_EPISODES} eps, >={MIN_UTTS} utts) {len(gallery):,}")
    print(f"  impostors (1 episode)     {len(imp):,}")
    print()
    pos_durs = sorted(d for v in gallery.values() for _, d in v["positive"])
    print(f"  enrolment clips           {n_enrol:,}")
    print(f"  positive trials           {n_pos:,}  (held-out episodes, longest first)")
    print(f"  impostor trials           {n_imp:,}")
    if pos_durs:
        print(f"  trial duration            p50 {pos_durs[len(pos_durs) // 2]:.2f}s  "
              f"max {pos_durs[-1]:.2f}s  >=3s {100 * sum(d >= 3 for d in pos_durs) / len(pos_durs):.0f}%")
    print(f"  embeddings to compute     {n_enrol + n_pos + n_imp:,}")
    print(f"\n  gallery covers {covered:,} / {len(rows):,} tai8 clips "
          f"({100 * covered / len(rows):.1f}%)")

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"gallery": gallery, "impostors": imp}, handle, ensure_ascii=False)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
