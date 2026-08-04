#!/usr/bin/env python3
"""
Bracket the word-frequency threshold for correct Taiwanese pronunciation.

Listening showed 今天 (1,545 occurrences in what v3b trained on) comes out right
while 天氣 (84) comes out wrong, even though 天 and 氣 are individually very
common. The mapping being learned is Mandarin-word to Taiwanese-word, so what
matters is how often the *word* recurs — and the usable threshold is somewhere in
an 18x range that nobody has measured.

Each sentence isolates one target word; everything around it is high-frequency so
the only thing varying is the target. Reference clip and seed are fixed, so the
only difference between takes is the word itself.

Listen to the target word alone and mark right or wrong. Where the answers flip
is the threshold, and that number decides how much data the next round needs —
guessing 300 versus 1,000 changes the requirement threefold.

  RUN=v3b python deploy/taipei1/08_word_threshold.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/mnt/shared/p06/indextts2"))
CKPT = Path(os.environ.get("CKPT", ROOT / "checkpoints"))
REFS = Path(os.environ.get("REFS_DIR", ROOT / "refs"))
RUN_DIR = Path(os.environ["RUN_DIR"])
PRUNED = Path(os.environ.get("CKPT_PRUNED") or RUN_DIR / "pruned" / "gpt.pth")
OUT = Path(os.environ.get("OUT_DIR") or RUN_DIR / "eval" / "word_threshold")
SEED = int(os.environ.get("SEED", "1234"))
REF = os.environ.get("REF", "ref_drama1_020_5.8s.wav")

# (target word, occurrences in the 150,840 sentences v3b saw, sentence)
CASES = [
    ("天氣", 84, "我今天看到外面的天氣"),
    ("出現", 214, "我今天看到他突然出現"),
    ("任何", 403, "我今天沒有看到任何人"),
    ("喜歡", 685, "我今天真的很喜歡他"),
    ("剛才", 1028, "我剛才有看到他們"),
    ("今天", 1545, "我今天有看到他們"),
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
    print(f"seed      : {SEED}\n")

    tts = IndexTTS2(
        cfg_path=str(CKPT / "config.yaml"),
        model_dir=str(CKPT),
        gpt_checkpoint_path=str(PRUNED),
        bpe_model_path=str(CKPT / "bpe.model"),
        use_fp16=True,
        use_deepspeed=False,
    )

    rows = []
    for word, count, text in CASES:
        # Reseed per utterance so each one is independent of generation order.
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        out = OUT / f"{count:05d}__{word}.wav"
        tts.infer(
            spk_audio_prompt=str(ref), text=text, output_path=str(out), verbose=False
        )
        rows.append((out.name, word, count, text))
        print(f"  {out.name}  目標「{word}」({count} 次)  {text}")

    index = OUT / "index.tsv"
    with index.open("w", encoding="utf-8") as handle:
        handle.write("file\ttarget_word\toccurrences\ttext\tcorrect\n")
        for name, word, count, text in rows:
            handle.write(f"{name}\t{word}\t{count}\t{text}\t\n")
    print(f"\nwrote {len(rows)} files and {index}")
    print("檔名前綴就是出現次數，依序播放，在 index.tsv 的 correct 欄填 y/n。")


if __name__ == "__main__":
    main()
