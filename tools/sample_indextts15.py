#!/usr/bin/env python3
"""Generate listening samples from the finetuned IndexTTS 1.5 model.

The 1.5 checkpoint cannot go into the local 8123 service: that stack loads
IndexTTS2, whose GPT, tokenizer and vocoder path differ. So the samples are
produced here and pulled down as audio instead.

The sentences are the ones already used to judge v4, so the two can be compared
directly. Which reference clip is used matters as much as the text: the earlier
1.5 run cloned nothing, and the point of this one is whether fixing where the
conditioning clip comes from restored that. Each sentence is therefore rendered
from a Taiwanese reference and from a Mandarin one.

  GPT_PATH=... OUT_DIR=... python tools/sample_indextts15.py
"""

from __future__ import annotations

import os
import sys

ROOT15 = os.environ.get("ROOT15", "/mnt/shared/p06/indextts15")
GPT_PATH = os.environ.get(
    "GPT_PATH", f"{ROOT15}/repo/finetune_models/checkpoints/gpt_epoch_4.pth"
)
CKPT_DIR = os.environ.get("CKPT_DIR", f"{ROOT15}/repo/checkpoints")
CFG = os.environ.get("CFG", f"{CKPT_DIR}/config.yaml")
REFS = os.environ.get("REFS_DIR", "/mnt/shared/p06/indextts2/refs")
OUT_DIR = os.environ.get("OUT_DIR", f"{ROOT15}/eval")
SEED = int(os.environ.get("SEED", "1234"))

# (label, what to listen for, text) — the same probes v4 was judged on.
CASES = [
    ("baseline", "整句（v4 幾乎全對）", "你不要再騙我了，我已經知道所有的事情"),
    ("tianqi", "天氣（v4 讀錯）", "我今天看到外面的天氣"),
    ("gangcai", "剛才（證據充足）", "我剛才有看到他們"),
    ("news", "多雲到晴（語料沒有）", "今天各地天氣多雲到晴"),
    ("classical", "東漢末年（文言文）", "東漢末年，天下大亂"),
]

# Cloning is the question this run exists to answer, so both a Taiwanese and a
# Mandarin reference are rendered: the failure being tested for showed up as the
# output ignoring whichever clip it was given.
REF_FILES = [
    ("drama1_020", "ref_drama1_020_5.8s.wav"),
    ("hayley", "ref_hayley_5s.wav"),
]


def main() -> None:
    import random

    import numpy as np
    import torch

    from indextts.infer import IndexTTS

    for path in (GPT_PATH, CFG):
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"  gpt   : {GPT_PATH}")
    print(f"  config: {CFG}")
    print(f"  seed  : {SEED}\n", flush=True)

    tts = IndexTTS(cfg_path=CFG, model_dir=CKPT_DIR, gpt_path=GPT_PATH, is_fp16=True)

    rows = []
    for ref_label, ref_name in REF_FILES:
        ref = os.path.join(REFS, ref_name)
        if not os.path.isfile(ref):
            print(f"  [skip] no {ref}")
            continue
        for case, listen_for, text in CASES:
            # Reseed per utterance so a take does not depend on generation order.
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            out = os.path.join(OUT_DIR, f"{ref_label}__{case}.wav")
            try:
                tts.infer(audio_prompt=ref, text=text, output_path=out, verbose=False)
            except Exception as exc:
                print(f"  [FAIL] {ref_label}/{case}: {type(exc).__name__}: {exc}")
                continue
            rows.append((os.path.basename(out), ref_name, listen_for, text))
            print(f"  {os.path.basename(out):32} 聽「{listen_for}」", flush=True)

    index = os.path.join(OUT_DIR, "index.tsv")
    with open(index, "w", encoding="utf-8") as handle:
        handle.write("file\treference\tlisten_for\ttext\tpronunciation_ok\tvoice_matches_ref\n")
        for name, ref, listen_for, text in rows:
            handle.write(f"{name}\t{ref}\t{listen_for}\t{text}\t\t\n")
    print(f"\n  wrote {len(rows)} files and {index}")
    print("  兩欄要分開判斷：pronunciation_ok 是台語念得對不對，")
    print("  voice_matches_ref 是音色像不像參考音檔（這次要驗證的重點）。")


if __name__ == "__main__":
    main()
