#!/usr/bin/env python3
"""Turn the index_tts manifests into audio_list parts for the IndexTTS 1.5 tools.

Same corpora v4 trained on, so the two runs differ only in architecture.

The 1.5 extraction tool wants `audio_list_part_N.txt`, one line per clip. Two
columns is what it shipped with; a third naming the speaker is added by
patch_indextts15.py, because without it a part looks like a single speaker and
the trainer draws each conditioning clip from whoever else lands in the same
part — the defect that cost the earlier 1.5 run its cloning.

Speakers are kept whole within a part. The conditioning sampler picks another
clip by the same speaker from the same manifest, so splitting a speaker across
parts would shrink the pool it can draw from for no benefit.

  OUT_DIR=... PARTS=4 python tools/build_indextts15_audio_lists.py
"""

from __future__ import annotations

import collections
import json
import os

DATASET = os.environ.get("DATASET", "/mnt/shared/p06/dataset202607_1")
CORPORA = os.environ.get("CORPORA", "tai8 naer").split()
SPLITS = os.environ.get("SPLITS", "train").split()
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/shared/p06/indextts15/finetune_data/audio_list")
PARTS = int(os.environ.get("PARTS", "4"))
MIN_DUR = float(os.environ.get("MIN_DUR", "1.0"))
MAX_DUR = float(os.environ.get("MAX_DUR", "15.0"))


def main() -> None:
    by_speaker: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    missing = skipped = 0

    for corpus in CORPORA:
        manifest_dir = f"{DATASET}/{corpus}/manifests/index_tts"
        for split in SPLITS:
            path = f"{manifest_dir}/{split}_manifest.jsonl"
            if not os.path.isfile(path):
                print(f"  [skip] no {path}")
                continue
            n = 0
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                if not MIN_DUR <= d.get("duration", 0) <= MAX_DUR:
                    skipped += 1
                    continue
                # Manifest paths are relative to the manifest directory.
                audio = os.path.normpath(os.path.join(manifest_dir, d["audio"]))
                if not os.path.isfile(audio):
                    missing += 1
                    continue
                text = d["text"].replace("\t", " ").strip()
                if not text:
                    skipped += 1
                    continue
                # Namespace the speaker so two corpora cannot collide on a label.
                by_speaker[f"{corpus}_{d['speaker']}"].append((audio, text))
                n += 1
            print(f"  {corpus}/{split}: {n:,} clips")

    if missing:
        print(f"  [warn] {missing:,} clips missing on disk")
    if skipped:
        print(f"  [warn] {skipped:,} clips outside {MIN_DUR}-{MAX_DUR}s or with empty text")

    # Greedy longest-first packing keeps the parts close in size, which matters
    # because one extraction process is launched per part.
    order = sorted(by_speaker, key=lambda s: -len(by_speaker[s]))
    buckets: list[list[str]] = [[] for _ in range(PARTS)]
    counts = [0] * PARTS
    for spk in order:
        i = counts.index(min(counts))
        buckets[i].append(spk)
        counts[i] += len(by_speaker[spk])

    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for i, spks in enumerate(buckets):
        path = f"{OUT_DIR}/audio_list_part_{i}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            for spk in spks:
                for audio, text in by_speaker[spk]:
                    handle.write(f"{audio}\t{text}\t{spk}\n")
                    total += 1
        print(f"  part {i}: {counts[i]:,} clips, {len(spks):,} speakers -> {path}")

    print(f"\n  {total:,} clips from {len(by_speaker):,} speakers into {PARTS} parts")


if __name__ == "__main__":
    main()
