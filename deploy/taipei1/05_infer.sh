#!/usr/bin/env bash
# Stage 5: synthesise with the finetuned checkpoint.
#
#   CKPT_PTH=/mnt/shared/p06/indextts2/runs/tai8_v1/model_step23560.pth \
#   PROMPT_WAV=/path/to/reference.wav \
#   TEXT='你今天吃飽了嗎' \
#
# TEXT is written in *Mandarin* Han characters, the same way the training
# transcripts were. The model learns the mapping from Mandarin orthography to
# Taiwanese pronunciation, so there is no need — and no reason — to write
# Taiwanese Hanji (今仔日, 食飽未) on the input side.
#   srun --ntasks=1 -p p06 --gres=gpu:h100:1 \
#     --container-image $ROOT/images/indextts2-vllm.sqsh \
#     --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
#     bash $ROOT/repo/deploy/taipei1/05_infer.sh
#
# INDEXTTS_ZH_T2S is exported by env.sh: inference must fold Traditional to
# Simplified exactly as preprocessing did, or the model sees text it never
# trained on.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO"

CKPT_PTH=${CKPT_PTH:-$ROOT/runs/tai8_v1/model_step23560.pth}
OUT_DIR=${OUT_DIR:-$ROOT/outputs}
TEXT=${TEXT:-請你明天早上來這裡找我}
PROMPT_WAV=${PROMPT_WAV:-}

mkdir -p "$OUT_DIR"

# Training checkpoints carry optimiser + scheduler state (~7.7GB). Strip it once
# so inference loads a plain weight file.
PRUNED=$OUT_DIR/$(basename "${CKPT_PTH%.pth}")_pruned.pth
if [[ ! -f "$PRUNED" ]]; then
    echo "=== pruning $(basename "$CKPT_PTH") -> $(basename "$PRUNED") ==="
    python tools/prune_gpt_checkpoint.py --input "$CKPT_PTH" --output "$PRUNED"
fi

# Fall back to the longest clip of an arbitrary training speaker as the voice
# reference, so the script runs with no arguments.
if [[ -z "$PROMPT_WAV" ]]; then
    PROMPT_WAV=$(python - <<'PY'
import json, os
manifest = os.path.expandvars("$WORK/shards/shard0/train_manifest.jsonl")
best = None
with open(manifest, encoding="utf-8") as handle:
    for i, line in enumerate(handle):
        if i > 20000:
            break
        r = json.loads(line)
        if best is None or (r.get("duration") or 0) > (best.get("duration") or 0):
            best = r
print(best["audio"])
PY
)
    echo "=== auto-selected prompt: $PROMPT_WAV ==="
fi

python - <<PY
import os
from indextts.infer_v2_modded import IndexTTS2

tts = IndexTTS2(
    cfg_path="$CKPT/config.yaml",
    model_dir="$CKPT",
    gpt_checkpoint_path="$PRUNED",
    bpe_model_path="$CKPT/bpe.model",
    use_fp16=True,
)
out = os.path.join("$OUT_DIR", "sample.wav")
tts.infer(
    spk_audio_prompt="$PROMPT_WAV",
    text="""$TEXT""",
    output_path=out,
    verbose=True,
)
print("wrote", out)
PY
