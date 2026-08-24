#!/usr/bin/env python3
"""Annotate Han text with Tâi-lô, in the one format IndexTTS's frontend survives.

The pronunciation problem this is aimed at is coverage: a character the corpus
never showed the model gets read as whatever Mandarin habit is nearest. Teaching
the model to read a romanisation turns that open-ended problem into a lookup —
any character whose reading can be found can be pinned.

Two things about IndexTTS decide the output format, and both were measured
rather than assumed:

  * Tone digits must not be followed by a space. WeTextProcessing expands
    units, so `tsuan2 khian5` becomes `tsuan两千小时ian五` — it read `2 kh` as
    two kilohours. Joining every syllable of an annotation with hyphens avoids
    it: `gik8-tsuan2-khian5-khun1` survives intact.
  * The tone digit itself becomes a Chinese numeral in mixed text (`khian5` ->
    `khian五`) but an English word in pure ASCII (`▁FIVE`). Training data and
    inference input therefore have to use the same mix, or the model is shown
    two encodings of one tone.

Lookup is word-first. Per-character assembly gets polyphones wrong — 乾坤 comes
out as kuann1-khun1, the 乾燥 reading, where the word entry gives khian5-khun.
A word with no entry is left unannotated rather than guessed, because a wrong
reading is worse than the model's own guess.

  python tools/tailo_annotate.py --text '他準備逆轉乾坤'
  python tools/tailo_annotate.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata

CHHOETAIGI = os.environ.get(
    "CHHOETAIGI", "/mnt/nas/cosyvoice_taipei_1_no5/chhoetaigi")
# Two patterns, because one character class cannot do both jobs: HAN.fullmatch
# on a multi-character headword never matches, so a single pattern silently
# reduced the lexicon to single characters — and per-character lookup is
# exactly the failure this tool exists to avoid.
HAN_CHAR = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
# Tâi-lô with diacritics, for reading the dictionaries' unicode columns.
TONE_MARKS = {0x0301: "2", 0x0300: "3", 0x0302: "5", 0x0304: "7", 0x030D: "8",
              0x030B: "9"}


def to_numeral_tone(text: str) -> str:
    """Diacritic Tâi-lô to numeral tone.

    The 8th tone is a combining vertical line (U+030D) with no precomposed
    form, so it has to be handled on the decomposed string.
    """
    out = []
    for syllable in re.split(r"([ \-]+)", text):
        if not syllable or re.fullmatch(r"[ \-]+", syllable):
            out.append(syllable)
            continue
        decomposed = unicodedata.normalize("NFD", syllable)
        tone = ""
        letters = []
        for ch in decomposed:
            if ord(ch) in TONE_MARKS:
                tone = TONE_MARKS[ord(ch)]
            elif unicodedata.combining(ch):
                continue
            else:
                letters.append(ch)
        body = "".join(letters)
        # Trailing punctuation has to come back out after the tone digit, or an
        # ellipsis swallows it: "lō..." became "lo...7" instead of "lo7...".
        core = body.rstrip(".,!?;:\"')]}…")
        trail = body[len(core):]
        if not core:
            out.append(body)
            continue
        if core[-1].isdigit():
            # Already numeral tone. Idempotent, so a mixed-notation source or a
            # second pass cannot turn Gua2 into Gua21.
            out.append(core + trail)
            continue
        if not tone:
            # Unmarked syllables are tone 1, or tone 4/8 when they end in a stop.
            tone = "4" if core[-1] in "ptkh" else "1"
        out.append(core + tone + trail)
    return "".join(out)


def load_lexicon(directory: str = CHHOETAIGI) -> dict[str, str]:
    """Han word -> Tâi-lô in numeral tone, longest entries winning ties."""
    lexicon: dict[str, str] = {}
    # HanLoTaibunKip only. The obvious idea is to also key on HoaBun, the
    # Mandarin gloss, since Mandarin is what gets typed here — but that column
    # is a translation, not a reading. Keying on it turns 茅廬 into tshau2-tshu3
    # (草厝) and 今天 into tsit-tsun7 (這陣): the Taiwanese word for the concept,
    # not how those characters are read. That is the 華語→台文 rewrite this
    # project has ruled out, and annotation is supposed to leave every character
    # exactly where it is.
    sources = [
        ("ChhoeTaigi_TaihoaSoanntengTuichiautian.csv", ("HanLoTaibunKip",), "KipInput"),
        ("ChhoeTaigi_iTaigiHoataiTuichiautian.csv", ("HanLoTaibunKip",), "KipInput"),
    ]
    for filename, han_cols, kip_col in sources:
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                kip = (row.get(kip_col) or "").strip().split("/")[0].strip()
                if not kip:
                    continue
                for han_col in han_cols:
                    han = (row.get(han_col) or "").strip()
                    # Entries still carrying Latin are Han-Lo mixed spellings,
                    # not a word that can be looked up from typed text.
                    if not han or " " in han or not HAN.fullmatch(han):
                        continue
                    reading = normalise_kip(kip)
                    # A reading has to have one syllable per character, or it
                    # does not belong to this headword: the dictionaries carry
                    # entries where the two columns describe different forms,
                    # and they surface as 三顧 -> koo3 or 撫鬚 -> tshiu1.
                    if len(reading.split("-")) != len(han):
                        continue
                    lexicon.setdefault(han, reading)
    return lexicon


def normalise_kip(kip: str) -> str:
    """One annotation, hyphen-joined, no spaces — see the module docstring."""
    kip = to_numeral_tone(kip) if not re.search(r"\d", kip) else kip
    kip = re.sub(r"\s+", "-", kip.strip())
    kip = re.sub(r"-{2,}", "-", kip)
    return kip


def annotate(text: str, lexicon: dict[str, str], max_word: int = 4,
             only: set[str] | None = None,
             substitute: bool = True) -> tuple[str, int]:
    """Insert readings after the words that have one.

    Longest match first, so 乾坤 is found before 乾. `only` restricts annotation
    to a given set of words, which is how training data is built with a
    controlled share of the sentence annotated.
    """
    out = []
    i = 0
    count = 0
    while i < len(text):
        if not HAN_CHAR.match(text[i]):
            out.append(text[i])
            i += 1
            continue
        for length in range(min(max_word, len(text) - i), 0, -1):
            word = text[i:i + length]
            reading = lexicon.get(word)
            if reading and (only is None or word in only):
                # Substitution, not an appended gloss. Appending leaves the
                # binding ambiguous — in "三顧 koo3" nothing says whether koo3
                # belongs to 顧 or to 三顧 — and substitution is the form the
                # CosyVoice3 run actually verified.
                out.append(f" {reading} " if substitute else f"{word} {reading} ")
                i += length
                count += 1
                break
        else:
            out.append(text[i])
            i += 1
    text_out = re.sub(r"\s+", " ", "".join(out)).strip()
    # Two annotations side by side leave a tone digit facing a letter across a
    # space, and WeTextProcessing reads that as a measurement: "siann1 si7"
    # became "siann一秒i七". Joining them closes the gap; a space next to a Han
    # character is harmless and stays.
    text_out = re.sub(r"(\d)\s+([A-Za-z])", r"\1-\2", text_out)
    return text_out, count


def rare_words(text: str, lexicon: dict[str, str], freq: dict[str, int],
               max_freq: int = 30, max_word: int = 4) -> tuple[set[str], list[str]]:
    """Which words in this text the corpus has too little of to have taught.

    Returns the ones worth annotating and the ones that need it but have no
    dictionary entry — the second list is the ceiling on what annotation can do,
    and it is not empty: 廬 is in neither ChhoeTaigi dictionary, so 三顧茅廬
    cannot be pinned however the model is trained.
    """
    pick, uncovered = set(), []
    i = 0
    while i < len(text):
        if not HAN_CHAR.match(text[i]):
            i += 1
            continue
        for length in range(min(max_word, len(text) - i), 0, -1):
            word = text[i:i + length]
            if word in lexicon and freq.get(word, 0) <= max_freq:
                pick.add(word)
                i += length
                break
        else:
            # Nothing in the lexicon matched here; if the single character is
            # also rare, annotation cannot reach it.
            if freq.get(text[i], 0) <= max_freq and text[i] not in lexicon:
                uncovered.append(text[i])
            i += 1
    return pick, uncovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--dict", default=CHHOETAIGI)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--rare-only", action="store_true",
                        help="annotate only words the training corpus barely has")
    parser.add_argument("--freq", default="work/tailo/word_freq.json")
    parser.add_argument("--max-freq", type=int, default=30)
    args = parser.parse_args()

    lexicon = load_lexicon(args.dict)
    print(f"  詞條 {len(lexicon):,}", file=sys.stderr)

    if args.selftest:
        for probe in ["他準備逆轉乾坤", "三顧茅廬", "老翁撫鬚而笑",
                      "從前有個樵夫", "我今天看到外面的天氣"]:
            sub, n = annotate(probe, lexicon)
            gloss, _ = annotate(probe, lexicon, substitute=False)
            print(f"  {probe}   ({n} 個詞標到)")
            print(f"    替換 -> {sub}")
            print(f"    對照 -> {gloss}")
        return

    if args.text:
        if args.rare_only:
            with open(args.freq, encoding="utf-8") as handle:
                freq = json.load(handle)
            pick, uncovered = rare_words(args.text, lexicon, freq, args.max_freq)
            annotated, n = annotate(args.text, lexicon, only=pick)
            print(annotated)
            print(f"  標了 {n} 個罕見詞: {' '.join(sorted(pick)) or '(無)'}", file=sys.stderr)
            if uncovered:
                print(f"  罕見但字典查無，標不到: {' '.join(uncovered)}", file=sys.stderr)
        else:
            annotated, n = annotate(args.text, lexicon)
            print(annotated)


if __name__ == "__main__":
    main()
