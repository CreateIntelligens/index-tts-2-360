#!/usr/bin/env python3
"""
Two listening grids, aimed at two specific questions.

Set A "voice"  — one text, every reference voice, both models.
    Does the finetuned model reproduce the reference speaker at all? The first
    round suggested it does not: the base model tracked the reference's timbre
    and slow delivery while the finetuned one did not.

Set B "domain" — one reference, texts from several domains, both models.
    Does it still speak Taiwanese outside prime-time-drama dialogue, or fall
    back to Mandarin on news / weather / instructional wording?

Reference clips are read from REFS_DIR rather than the corpus, so this runs even
when the dataset mount is unreadable.

  REFS_DIR=... CKPT_PRUNED=... python deploy/taipei1/07_eval_matrix.py
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/mnt/shared/p06/indextts2"))
CKPT = Path(os.environ.get("CKPT", ROOT / "checkpoints"))
REFS = Path(os.environ.get("REFS_DIR", ROOT / "refs"))
OUT = Path(os.environ.get("OUT_DIR", ROOT / "outputs/eval"))
PRUNED = Path(os.environ["CKPT_PRUNED"])

# Everything below is Mandarin orthography; the model is what turns it into
# Taiwanese. Lengths are deliberately longer than the first round's 7-21 chars.
VOICE_TEXT = "你不要再騙我了，我已經知道所有的事情，你到底把錢藏到哪裡去了"

DOMAIN_TEXTS = [
    ("drama", "你不要再騙我了，我已經知道所有的事情，你到底把錢藏到哪裡去了"),
    ("news", "行政院今天下午召開記者會，宣布明年將投入三百億元推動人工智慧產業發展"),
    ("weather", "明天各地天氣多雲到晴，北部地區清晨較涼，白天高溫可達攝氏三十度，出門記得攜帶雨具"),
    ("daily", "我昨天去菜市場買了一些青菜和魚，老闆說今天的魚特別新鮮，價格也比較便宜"),
    ("manual", "請先按下電源按鈕，等待螢幕出現藍色畫面之後，再插入記憶卡並選擇安裝選項"),
]

DOMAIN_REF = os.environ.get("DOMAIN_REF", "ref_drama1_020_5.8s.wav")


def load(gpt_path: Path):
    from indextts.infer_v2_modded import IndexTTS2

    return IndexTTS2(
        cfg_path=str(CKPT / "config.yaml"),
        model_dir=str(CKPT),
        gpt_checkpoint_path=str(gpt_path),
        bpe_model_path=str(CKPT / "bpe.model"),
        use_fp16=True,
        use_deepspeed=False,
    )


def release(tts) -> None:
    import torch

    del tts
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    refs = sorted(p for p in REFS.glob("*.wav") if p.is_file())
    if not refs:
        raise SystemExit(f"no reference wavs in {REFS}")
    print("references:")
    for ref in refs:
        print(f"  {ref.name}")

    domain_ref = REFS / DOMAIN_REF
    if not domain_ref.is_file():
        domain_ref = refs[0]
    print(f"domain-grid reference: {domain_ref.name}\n")

    manifest_lines = []

    for tag, gpt_path in (("base", CKPT / "gpt.pth"), ("tai8", PRUNED)):
        print(f"===== {tag} =====")
        tts = load(gpt_path)

        # Set A: every voice, one text.
        for ref in refs:
            out = OUT / f"voice__{ref.stem}__{tag}.wav"
            tts.infer(spk_audio_prompt=str(ref), text=VOICE_TEXT, output_path=str(out), verbose=False)
            manifest_lines.append(f"{out.name}\t{ref.name}\t{tag}\t{VOICE_TEXT}")
            print(f"  {out.name}")

        # Set B: one voice, several domains.
        for name, text in DOMAIN_TEXTS:
            out = OUT / f"domain__{name}__{tag}.wav"
            tts.infer(spk_audio_prompt=str(domain_ref), text=text, output_path=str(out), verbose=False)
            manifest_lines.append(f"{out.name}\t{domain_ref.name}\t{tag}\t{text}")
            print(f"  {out.name}")

        release(tts)

    index = OUT / "index.tsv"
    index.write_text(
        "file\treference\tmodel\ttext\n" + "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(manifest_lines)} files and {index}")


if __name__ == "__main__":
    main()
