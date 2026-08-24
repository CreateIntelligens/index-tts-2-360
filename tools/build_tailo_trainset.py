#!/usr/bin/env python3
"""Build the dual-script training set that teaches IndexTTS to read Tâi-lô.

SuíSiann gives 3,467 clips with both a Han and a Tâi-lô transcription of the
same audio, which is the only clean source of that pairing available here. What
it does not give is a word-level alignment between the two, and the training
format has to be word-level: at inference only the handful of words a dictionary
covers get annotated, never the whole sentence.

The alignment is done by verification rather than by parsing. A candidate
reading from ChhoeTaigi is only used if it actually occurs in that clip's own
Tâi-lô transcription — so a wrong dictionary sense, or a word the speaker read
differently, is dropped instead of teaching the model a reading the audio does
not contain.

Three rows come out of each clip:

  hanji     the Han text unchanged, so the ability already there is rehearsed
  mixed     some words replaced by their reading — the inference-time shape
  tailo     the whole sentence romanised, which is what makes the letters mean
            sounds at all

Only `mixed` matches how this gets used, but a model shown only `mixed` has no
reason to read the romanised span as anything but noise; `tailo` supplies the
grounding and `hanji` stops the Han side drifting.

  python tools/build_tailo_trainset.py --out work/tailo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tailo_annotate import (HAN_CHAR, annotate, load_lexicon,  # noqa: E402
                            normalise_kip, to_numeral_tone)

SUISIANN = os.environ.get(
    "SUISIANN", "/mnt/nas/cosyvoice_taipei_1_no5/suisiann/0.2.1")


def verified_words(han: str, tailo: str, lexicon: dict[str, str],
                   max_word: int = 4) -> set[str]:
    """Words whose dictionary reading really is in this clip's romanisation."""
    reference = re.sub(r"[^a-z0-9]", "", to_numeral_tone(tailo).lower())
    found = set()
    i = 0
    while i < len(han):
        if not HAN_CHAR.match(han[i]):
            i += 1
            continue
        for length in range(min(max_word, len(han) - i), 0, -1):
            word = han[i:i + length]
            reading = lexicon.get(word)
            if reading and re.sub(r"[^a-z0-9]", "", reading.lower()) in reference:
                found.add(word)
                i += length
                break
        else:
            i += 1
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="work/tailo")
    parser.add_argument("--suisiann", default=SUISIANN)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--annotate-share", type=float, default=0.5,
                        help="fraction of verified words to replace in a mixed row")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lexicon = load_lexicon()
    print(f"  詞條 {len(lexicon):,}")

    rows = list(csv.DictReader(open(f"{args.suisiann}/SuiSiann.csv", encoding="utf-8")))
    os.makedirs(args.out, exist_ok=True)

    out = []
    stats = {"clips": 0, "verified": 0, "candidates": 0, "mixed": 0}
    for row in rows:
        han = row["漢字"].strip()
        tailo = row["羅馬字"].strip()
        audio = os.path.join(args.suisiann, row["音檔"])
        if not han or not tailo or not os.path.isfile(audio):
            continue
        stats["clips"] += 1

        _, candidates = annotate(han, lexicon)
        stats["candidates"] += candidates
        good = verified_words(han, tailo, lexicon)
        stats["verified"] += len(good)

        out.append({"audio": audio, "text": han, "kind": "hanji"})
        out.append({"audio": audio, "text": normalise_kip(to_numeral_tone(tailo)),
                    "kind": "tailo"})
        if good:
            chosen = {w for w in good if rng.random() < args.annotate_share} or {
                rng.choice(sorted(good))}
            mixed, n = annotate(han, lexicon, only=chosen)
            if n:
                out.append({"audio": audio, "text": mixed, "kind": "mixed"})
                stats["mixed"] += 1

    path = os.path.join(args.out, "trainset.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for item in out:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  clips              {stats['clips']:,}")
    print(f"  字典命中的詞       {stats['candidates']:,}")
    print(f"  其中讀音對得上的   {stats['verified']:,} "
          f"({100 * stats['verified'] / max(stats['candidates'], 1):.1f}%)")
    print(f"  產生 mixed 的 clip {stats['mixed']:,}")
    print(f"  總列數             {len(out):,} -> {path}")

    for kind in ("hanji", "tailo", "mixed"):
        sample = next((r for r in out if r["kind"] == kind), None)
        if sample:
            print(f"\n  [{kind}] {sample['text'][:78]}")


if __name__ == "__main__":
    main()
