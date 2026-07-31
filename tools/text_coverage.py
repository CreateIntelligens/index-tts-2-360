#!/usr/bin/env python3
"""
Score a text against the training corpus before spending GPU time on it.

Pronunciation is learned at two levels and both can fail independently:

  characters  a character the corpus never contains has no Taiwanese evidence at
              all, so the model falls back to the base model's Mandarin.
  words       Taiwanese is not the concatenation of per-character readings. The
              subtitles write Mandarin (丈夫) where the actors say Taiwanese
              (翁/ang), so the mapping is learned per word. A compound whose
              characters are common can still be unreliable: 明天 appears 210
              times against 今天's 2,124, and 明天 is the one that comes out
              wrong. Readings observed in the wild for 丈夫 — ang-sai, ang-si,
              ang-fu — show the model oscillating between the word-level mapping
              and a literal character reading when evidence is thin.

Example:
    python tools/text_coverage.py --text '明天各地天氣多雲到晴，北部地區清晨較涼'
    python tools/text_coverage.py --file sentences.txt --manifest <corpus>.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HAN = re.compile(r"[一-鿿]")
HAN_RUN = re.compile(r"[一-鿿]+")

DEFAULT_MANIFESTS = [
    "/mnt/nas/dataset202607_1/tai8/manifests/index_tts/train_manifest.jsonl",
]

# Thresholds. The word figures come from what the corpus actually supports:
# 今天 (2,124) is reliable, 明天 (210) is not, so "hundreds" is not enough.
CHAR_RARE = 50
WORD_WEAK = 150
WORD_SOLID = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report corpus coverage for a text.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Text to score.")
    src.add_argument("--file", type=Path, help="File with one text per line.")
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        help="Corpus manifest JSONL (repeatable). Defaults to the tai8 train manifest.",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-word-len", type=int, default=4, help="Longest n-gram to score.")
    parser.add_argument("--top", type=int, default=20, help="How many risks to list.")
    return parser.parse_args()


def build_index(manifests: list[str], field: str, max_len: int):
    """Character counts plus n-gram counts up to max_len, over the whole corpus."""
    chars: collections.Counter = collections.Counter()
    grams: collections.Counter = collections.Counter()
    lines = 0
    for path in manifests:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get(field, "")
                lines += 1
                chars.update(HAN.findall(text))
                # n-grams over contiguous Han runs only; punctuation breaks words.
                for run in HAN_RUN.findall(text):
                    for n in range(2, max_len + 1):
                        for i in range(len(run) - n + 1):
                            grams[run[i : i + n]] += 1
    return chars, grams, lines


def score(text: str, chars, grams, max_len: int, top: int) -> None:
    cs = HAN.findall(text)
    if not cs:
        print("  (no Han characters)")
        return

    unseen = [c for c in cs if chars[c] == 0]
    rare = [(c, chars[c]) for c in cs if 0 < chars[c] < CHAR_RARE]
    risk = len(set(unseen)) + len(set(c for c, _ in rare))

    print(f"  字數 {len(cs)}   未見 {len(set(unseen))}   <{CHAR_RARE}次 {len(set(c for c,_ in rare))}"
          f"   高風險字比例 {(len(unseen) + len(rare)) / len(cs) * 100:.1f}%")
    if unseen:
        print(f"    未見字: {''.join(sorted(set(unseen)))}")
    if rare:
        shown = sorted(set(rare), key=lambda x: x[1])[:top]
        print(f"    低頻字: {'  '.join(f'{c}({v})' for c, v in shown)}")

    # Bigrams, not longest-match. Longest-match reported junk like 你不要再(182)
    # — a 4-gram that happens to recur — instead of the two-character units that
    # actually carry the Mandarin-word-to-Taiwanese-word mapping. Some bigrams
    # straddle a word boundary and are legitimately rare, so this is a weaker
    # signal than the character counts and is reported, not scored.
    bigrams = []
    for run in HAN_RUN.findall(text):
        for i in range(len(run) - 1):
            g = run[i : i + 2]
            bigrams.append((g, grams[g]))

    absent = sorted({g for g, c in bigrams if c == 0})
    thin = sorted({g: c for g, c in bigrams if 0 < c < WORD_WEAK}.items(), key=lambda x: x[1])
    if absent:
        print(f"    從未同時出現的字對 ({len(absent)}): {'  '.join(absent[:top])}")
    if thin:
        print(f"    字對證據薄弱 (<{WORD_WEAK}): {'  '.join(f'{g}({c})' for g, c in thin[:top])}")
        print("      （跨詞邊界的字對本來就少，這欄要自己判斷哪些是真的詞）")

    # The verdict keys off characters: an unseen character has no Taiwanese
    # evidence at all, which is the one failure mode that is certain.
    unseen_ratio = len(unseen) / len(cs)
    if unseen_ratio > 0.05:
        verdict = "高風險 — 預期大量退回華語"
    elif unseen:
        verdict = "中高風險 — 未見字附近會出錯"
    elif rare:
        verdict = "中風險 — 低頻字可能出錯"
    elif absent:
        verdict = "低-中風險 — 字都夠常見，但有沒見過的詞"
    else:
        verdict = "低風險"
    print(f"    → {verdict}")


def main() -> None:
    args = parse_args()
    manifests = args.manifests or DEFAULT_MANIFESTS
    for path in manifests:
        if not Path(path).is_file():
            sys.exit(f"manifest not found: {path}")

    print(f"building index from {len(manifests)} manifest(s)...", file=sys.stderr)
    chars, grams, lines = build_index(manifests, args.text_field, args.max_word_len)
    print(f"  {lines:,} 句, {len(chars):,} 相異字, {len(grams):,} 相異詞組\n", file=sys.stderr)

    texts = [args.text] if args.text else [
        t.strip() for t in args.file.read_text(encoding="utf-8").splitlines() if t.strip()
    ]
    for i, text in enumerate(texts, 1):
        preview = text if len(text) <= 40 else text[:40] + "…"
        print(f"[{i}] {preview}")
        score(text, chars, grams, args.max_word_len, args.top)
        print()


if __name__ == "__main__":
    main()
