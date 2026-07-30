#!/usr/bin/env python3
"""
Split manifests into N shards for parallel preprocessing, one shard per GPU.

Sharding is done *by speaker*, never by line: `tools/build_gpt_prompt_pairs.py`
pairs a target with another utterance from the same speaker, so splitting a
speaker across shards would silently destroy those pairs. Speakers are assigned
greedily (largest first) to whichever shard currently holds the fewest clips,
which keeps the shards close to equal in size despite the long tail.

Train and validation manifests are sharded with the *same* speaker assignment so
a speaker never straddles shards.

Example:
    python tools/shard_manifest.py \
        --train-manifest work/train.filtered.jsonl \
        --val-manifest   work/val.filtered.jsonl \
        --output-dir     work/shards --shards 4
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard manifests by speaker for parallel preprocessing.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Source training manifest (read-only).")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Optional validation manifest (read-only).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write shard subdirectories into.")
    parser.add_argument("--shards", type=int, required=True, help="Number of shards (typically one per GPU).")
    parser.add_argument("--speaker-field", type=str, default="speaker", help="Record field holding the speaker id.")
    return parser.parse_args()


def read_records(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be >= 1")

    train = read_records(args.train_manifest)
    val = read_records(args.val_manifest) if args.val_manifest else []
    if not train:
        raise RuntimeError(f"No entries in {args.train_manifest}")

    # Count each speaker across both splits so the balancing reflects real work.
    counts: collections.Counter = collections.Counter()
    for record in train + val:
        counts[record.get(args.speaker_field, "")] += 1

    # Greedy largest-first bin packing.
    loads = [0] * args.shards
    assignment: Dict[str, int] = {}
    for speaker, count in counts.most_common():
        target = min(range(args.shards), key=lambda i: loads[i])
        assignment[speaker] = target
        loads[target] += count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = [collections.Counter() for _ in range(args.shards)]

    for label, records in (("train", train), ("val", val)):
        if not records:
            continue
        handles = []
        for i in range(args.shards):
            shard_dir = args.output_dir / f"shard{i}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            handles.append((shard_dir / f"{label}_manifest.jsonl").open("w", encoding="utf-8"))
        try:
            for record in records:
                i = assignment[record.get(args.speaker_field, "")]
                handles[i].write(json.dumps(record, ensure_ascii=False) + "\n")
                written[i][label] += 1
        finally:
            for handle in handles:
                handle.close()

    print(f"[Done] {len(counts):,} speakers -> {args.shards} shards")
    for i in range(args.shards):
        parts = "  ".join(f"{k}={v:,}" for k, v in sorted(written[i].items()))
        speakers = sum(1 for s, sh in assignment.items() if sh == i)
        print(f"  shard{i}: speakers={speakers:5,}  {parts}")
    spread = (max(loads) - min(loads)) / max(1, sum(loads) / args.shards)
    print(f"[Info] load imbalance: {spread * 100:.2f}% (max-min relative to mean)")


if __name__ == "__main__":
    main()
