#!/usr/bin/env python3
"""The four-way test that decides whether the Tâi-lô training worked.

Loss cannot answer this. The question is whether an annotation can override the
model's own guess at a word it was never taught, and that only shows up in what
comes out of the speaker.

Four inputs per sentence, at the same seeds:

  tailo     the sentence fully romanised, none of it seen in training
  hanji     the same sentence in Han — the control, and the thing that must not
            have got worse
  mixed     Han with the rare words replaced by their reading, which is how this
            would actually be used
  plain     identical to hanji, generated again alongside mixed so the pair
            differs in nothing but the annotation

mixed against plain is the whole experiment. The CosyVoice3 run moved one
polyphone from 2 of 3 seeds correct to 5 of 5 that way; anything less than a
clear shift means the annotation is decoration.

  python tools/tailo_listen.py --models tailo_ep8,v5_ep8 --out demo_tailo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tailo_annotate import annotate, load_lexicon, rare_words  # noqa: E402

TTS = "http://localhost:8065"
ASR = "http://localhost:8068"

# Sentences chosen for the words the corpus does not have: 乾坤 0 occurrences,
# 樵夫 0, 茅 7, 廬 1. The last one is here as the known ceiling — 廬 is in
# neither dictionary, so no annotation can reach it.
SENTENCES = [
    "他準備逆轉乾坤",
    "從前有個樵夫每天上山砍柴",
    "老翁撫鬚而笑",
    "劉備三顧茅廬請諸葛亮",
    "書生連夜趕路來到一座破廟前",
]
SEEDS = [2, 7, 42, 123, 777]


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def synth(text: str, character: str, seed: int) -> bytes:
    request = urllib.request.Request(
        f"{TTS}/tts", data=json.dumps(
            {"text": text, "character": character, "seed": seed}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        return response.read()


def asr(wav: bytes, reference: str) -> dict:
    boundary = "----listen"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"reference\"\r\n\r\n{reference}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; "
             f"filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += wav + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{ASR}/transcribe", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    # Point this at the serving tree's static directory to listen in a browser;
    # relative by default so the script does not assume where that tree lives.
    parser.add_argument("--out", default="demo_tailo")
    parser.add_argument("--character", default="hayley")
    parser.add_argument("--freq", default="work/tailo/word_freq.json")
    args = parser.parse_args()

    lexicon = load_lexicon()
    with open(args.freq, encoding="utf-8") as handle:
        freq = json.load(handle)
    os.makedirs(args.out, exist_ok=True)

    # Built once so every model is asked exactly the same thing.
    probes = []
    for sentence in SENTENCES:
        pick, uncovered = rare_words(sentence, lexicon, freq)
        mixed, n = annotate(sentence, lexicon, only=pick)
        full, _ = annotate(sentence, lexicon)
        probes.append({"hanji": sentence, "mixed": mixed, "tailo": full,
                       "annotated": sorted(pick), "uncovered": uncovered,
                       "n": n})
        print(f"  {sentence}")
        print(f"    標音 {n} 個: {' '.join(sorted(pick)) or '(無)'}"
              f"{'   標不到: ' + ' '.join(uncovered) if uncovered else ''}")
        print(f"    混排 -> {mixed}")

    # Written as they arrive. A run that dies on the second model used to lose
    # the first model's results too — and switching can fail for reasons that
    # have nothing to do with the model, such as the VRAM residue that
    # accumulates across switches until the KV cache budget goes negative.
    stream_path = os.path.join(args.out, "rows.jsonl")
    stream = open(stream_path, "a", encoding="utf-8")
    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {model} ===", flush=True)
        result = post(f"{TTS}/model/switch", {"name": model})
        if result.get("status") != "ok":
            print(f"  [FAIL] {result}")
            continue
        print(f"  已切換 {result['seconds']}s", flush=True)

        for index, probe in enumerate(probes):
            for kind in ("tailo", "hanji", "mixed", "plain"):
                text = probe["hanji"] if kind == "plain" else probe[kind]
                for seed in SEEDS:
                    try:
                        wav = synth(text, args.character, seed)
                        scored = asr(wav, probe["hanji"])
                    except Exception as exc:
                        print(f"  [skip] {kind} s{seed}: {type(exc).__name__}")
                        continue
                    name = f"{model}__{index}__{kind}__s{seed}.wav"
                    with open(os.path.join(args.out, name), "wb") as handle:
                        handle.write(wav)
                    row = {"model": model, "sentence": probe["hanji"],
                           "kind": kind, "seed": seed, "input": text,
                           "cer": scored["cer"], "hyp": scored["text"],
                           "file": name}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
            print(f"  {index}. {probe['hanji'][:16]} 完成", flush=True)

    stream.close()
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"probes": probes, "rows": rows}, handle, ensure_ascii=False)
    print(f"\n  {len(rows)} 筆 -> {args.out}")


if __name__ == "__main__":
    main()
