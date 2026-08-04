#!/usr/bin/env python3
"""
Test whether an unknown word derails the rest of the utterance.

The word-frequency hypothesis broke. 天氣 has 84 occurrences in what v3b trained
on, and it comes out *correct* in 「我今天看到外面的天氣」but *wrong* in
「今天各地天氣多雲到晴」. Same word, same count, opposite outcomes — so what
matters is the context, not the count on its own.

The suspicion is contagion: generation is autoregressive, so a word the corpus
never contains derails the sequence and everything after it inherits the damage.
各地 (0 occurrences) sits before 天氣 in the failing sentence.

Each case keeps 天氣 as the probe and moves the unknown word around it:

  A  clean            no unknown word          expect correct
  B  unknown before   各地 ahead of 天氣        expect wrong, if contagion is real
  C  unknown after    各地 behind 天氣          correct => damage flows forward only
  D  unknown at start 各地 opens the sentence   wrong => the whole utterance derails
  E  original         the sentence that failed  the reference point

If C is correct and B is wrong, the fix is to eliminate zero-coverage words
rather than to push every word's count higher — a very different data strategy.

  RUN_DIR=<version dir> python deploy/taipei1/09_context_contagion.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/mnt/shared/p06/indextts2"))
CKPT = Path(os.environ.get("CKPT", ROOT / "checkpoints"))
REFS = Path(os.environ.get("REFS_DIR", ROOT / "refs"))
RUN_DIR = Path(os.environ["RUN_DIR"])
PRUNED = Path(os.environ.get("CKPT_PRUNED") or RUN_DIR / "pruned" / "gpt.pth")
OUT = Path(os.environ.get("OUT_DIR") or RUN_DIR / "eval" / "context_contagion")
SEED = int(os.environ.get("SEED", "1234"))
REF = os.environ.get("REF", "ref_drama1_020_5.8s.wav")

# (case, what to listen for, sentence)
CASES = [
    ("A_clean", "天氣", "我今天看到外面的天氣"),
    ("B_unknown_before", "天氣（前面有各地）", "我今天看到各地外面的天氣"),
    ("C_unknown_after", "天氣（後面才有各地）", "我今天看到外面的天氣各地"),
    ("D_unknown_first", "天氣（各地開頭）", "各地我今天看到外面的天氣"),
    ("E_original", "天氣（原始失敗句）", "今天各地天氣多雲到晴"),
]


def main() -> None:
    import random

    import numpy as np
    import torch

    from indextts.infer_v2_modded import IndexTTS2

    if not PRUNED.is_file():
        raise SystemExit(f"no pruned checkpoint at {PRUNED}")
    ref = REFS / REF
    if not ref.is_file():
        raise SystemExit(f"no reference clip at {ref}")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"model     : {PRUNED}")
    print(f"reference : {ref.name}")
    print(f"seed      : {SEED}")
    print("probe     : 天氣 (84 occurrences) in every case\n")

    tts = IndexTTS2(
        cfg_path=str(CKPT / "config.yaml"),
        model_dir=str(CKPT),
        gpt_checkpoint_path=str(PRUNED),
        bpe_model_path=str(CKPT / "bpe.model"),
        use_fp16=True,
        use_deepspeed=False,
    )

    rows = []
    for case, listen_for, text in CASES:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        out = OUT / f"{case}.wav"
        tts.infer(spk_audio_prompt=str(ref), text=text, output_path=str(out), verbose=False)
        rows.append((out.name, case, listen_for, text))
        print(f"  {out.name:24} 聽「{listen_for}」  {text}")

    index = OUT / "index.tsv"
    with index.open("w", encoding="utf-8") as handle:
        handle.write("file\tcase\tlisten_for\ttext\ttarget_correct\n")
        for name, case, listen_for, text in rows:
            handle.write(f"{name}\t{case}\t{listen_for}\t{text}\t\n")
    print(f"\nwrote {len(rows)} files and {index}")
    print("每一句都只聽「天氣」兩個字對不對，填進 index.tsv 的 target_correct 欄。")


if __name__ == "__main__":
    main()
