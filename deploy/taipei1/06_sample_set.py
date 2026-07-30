#!/usr/bin/env python3
"""
Generate a listening set: the same Mandarin sentences through the base checkpoint
and through a finetuned one, so the pair can be compared directly.

If the finetuning worked, the base model reads the text in Mandarin and the
finetuned model reads the same characters with Taiwanese pronunciation.

  CKPT_PRUNED=/path/to/pruned.pth python deploy/taipei1/06_sample_set.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/mnt/shared/p06/indextts2"))
CKPT = Path(os.environ.get("CKPT", ROOT / "checkpoints"))
WORK = Path(os.environ.get("WORK", ROOT / "work"))
OUT = Path(os.environ.get("OUT_DIR", ROOT / "outputs/samples"))
PRUNED = Path(os.environ["CKPT_PRUNED"])

# All Mandarin orthography. Nothing here is Taiwanese Hanji.
SENTENCES = [
    ("basic1", "你今天吃飽了嗎"),
    ("basic2", "請你明天早上來這裡找我"),
    ("longer", "我跟他說過這件事情真的很重要，你不要再問了"),
    # 後 folds onto 后 under t2s, so this probes the merge ambiguity.
    ("merge_hou", "後面那個人是我的朋友"),
    # 準 folds onto 准 - the one merge group that is not lopsided.
    ("merge_zhun", "這個時間不太準，你再確認一次"),
]


def pick_prompt() -> str:
    """Longest clip from a training speaker, so the timbre reference is solid."""
    manifest = WORK / "shards/shard0/train_manifest.jsonl"
    best = None
    with manifest.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i > 20000:
                break
            record = json.loads(line)
            if best is None or (record.get("duration") or 0) > (best.get("duration") or 0):
                best = record
    if best is None:
        sys.exit(f"no usable prompt in {manifest}")
    print(f"prompt: {best['audio']}  ({best['duration']:.2f}s, speaker {best['speaker']})")
    print(f"prompt transcript: {best.get('text','')}")
    return best["audio"]


def main() -> None:
    from indextts.infer_v2_modded import IndexTTS2

    OUT.mkdir(parents=True, exist_ok=True)
    prompt = pick_prompt()

    for tag, gpt_path in (("base", CKPT / "gpt.pth"), ("tai8", PRUNED)):
        print(f"\n===== {tag}: {gpt_path} =====")
        tts = IndexTTS2(
            cfg_path=str(CKPT / "config.yaml"),
            model_dir=str(CKPT),
            gpt_checkpoint_path=str(gpt_path),
            bpe_model_path=str(CKPT / "bpe.model"),
            use_fp16=True,
            use_deepspeed=False,
        )
        for name, text in SENTENCES:
            out = OUT / f"{name}__{tag}.wav"
            tts.infer(
                spk_audio_prompt=prompt,
                text=text,
                output_path=str(out),
                verbose=False,
            )
            print(f"  {out.name}  <- {text}")
        del tts
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nwrote {len(SENTENCES) * 2} files to {OUT}")


if __name__ == "__main__":
    main()
