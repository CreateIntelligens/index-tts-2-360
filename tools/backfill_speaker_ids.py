#!/usr/bin/env python3
"""Fill speaker_id into the metadata the IndexTTS 1.5 extractor produced.

The extractor rebuilds its per-clip dict twice — once in the dataset's
__getitem__ and once when the batch is reassembled — and neither carries a field
it was not written to expect, so a speaker threaded through the audio_list
arrives at the metadata writer as None. Joining afterwards on the audio path
gets the same result without touching three points in a pipeline whose output we
depend on.

The trainer needs this: its lazy sampler picks the conditioning clip from
whichever offsets share a manifest, and a manifest here is `audio_list_part_N`,
an arbitrary slice of the corpus. Without a real speaker on each row the prompt
is somebody else nearly every time, which teaches the model to disregard the
prompt — the failure the earlier 1.5 run showed.

  AUDIO_LIST_DIR=... PROCESSED_DIR=... python tools/backfill_speaker_ids.py
"""

from __future__ import annotations

import collections
import glob
import json
import os

AUDIO_LIST_DIR = os.environ.get(
    "AUDIO_LIST_DIR", "/mnt/shared/p06/indextts15/finetune_data/audio_list"
)
PROCESSED_DIR = os.environ.get(
    "PROCESSED_DIR", "/mnt/shared/p06/indextts15/finetune_data/processed_data"
)


def main() -> None:
    speaker_of: dict[str, str] = {}
    for path in sorted(glob.glob(f"{AUDIO_LIST_DIR}/audio_list_part_*.txt")):
        n = 0
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                speaker_of[parts[0]] = parts[2]
                n += 1
        print(f"  {os.path.basename(path)}: {n:,} rows")
    if not speaker_of:
        raise SystemExit(f"no 3-column audio_list rows under {AUDIO_LIST_DIR}")
    print(f"  {len(speaker_of):,} audio paths carry a speaker\n")

    targets = sorted(glob.glob(f"{PROCESSED_DIR}/*/metadata_*.jsonl"))
    if not targets:
        raise SystemExit(f"no metadata_*.jsonl under {PROCESSED_DIR}")

    total = filled = missed = 0
    per_file_speakers = collections.Counter()
    for path in targets:
        rows = []
        hit = miss = 0
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            spk = speaker_of.get(d.get("audio", ""))
            if spk:
                d["speaker_id"] = spk
                hit += 1
                per_file_speakers[path] += 0
            else:
                miss += 1
            rows.append(d)
        with open(path, "w", encoding="utf-8") as handle:
            for d in rows:
                handle.write(json.dumps(d, ensure_ascii=False) + "\n")
        distinct = len({d.get("speaker_id") for d in rows if d.get("speaker_id")})
        print(f"  {os.path.relpath(path, PROCESSED_DIR)}: {hit:,} filled, "
              f"{miss:,} unmatched, {distinct:,} speakers")
        total += len(rows)
        filled += hit
        missed += miss

    print(f"\n  {filled:,}/{total:,} rows carry a speaker_id")
    if missed:
        print(f"  [warn] {missed:,} rows unmatched — those fall back to "
              f"whole-manifest sampling")


if __name__ == "__main__":
    main()
