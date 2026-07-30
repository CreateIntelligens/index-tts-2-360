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


HAN = re.compile(r"[一-鿿]")

# The prompt sets timbre *and* prosody, so it has to be a well-aligned clip.
# Picking purely by duration lands on the corpus's worst tail: the 31 clips over
# 10s average 0.67 characters per second, i.e. mostly audio the transcript does
# not cover, which then skews the generated speaking rate.
MIN_PROMPT_S, MAX_PROMPT_S = 3.0, 8.0
MIN_RATE, MAX_RATE = 4.0, 6.5  # corpus median is 5.2 characters per second


def pick_prompt() -> str:
    """Longest *well-aligned* clip: long enough for timbre, sane speaking rate."""
    manifest = WORK / "shards/shard0/train_manifest.jsonl"
    candidates = []
    fallback = None
    with manifest.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i > 40000:
                break
            record = json.loads(line)
            duration = record.get("duration") or 0.0
            chars = len(HAN.findall(record.get("text", "")))
            if not duration or not chars:
                continue
            rate = chars / duration
            if fallback is None or duration > (fallback.get("duration") or 0):
                fallback = record
            if MIN_PROMPT_S <= duration <= MAX_PROMPT_S and MIN_RATE <= rate <= MAX_RATE:
                candidates.append((duration, rate, record))

    if candidates:
        candidates.sort(key=lambda item: -item[0])
        duration, rate, best = candidates[0]
        print(f"prompt: {best['audio']}")
        print(f"  {duration:.2f}s, {rate:.2f} chars/s, speaker {best['speaker']}")
        print(f"  transcript: {best.get('text','')}")
        print(f"  ({len(candidates):,} clips met the alignment filter)")
        return best["audio"]

    if fallback is None:
        sys.exit(f"no usable prompt in {manifest}")
    print(f"[Warn] no well-aligned clip found; falling back to {fallback['audio']}")
    return fallback["audio"]


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
