#!/usr/bin/env bash
# Runs inside the container: launches torchrun with one process per GPU.
#
# The GPU count comes from NUM_GPUS at call time, so the same script covers a
# 1-GPU smoke test and an N-GPU production run.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO"

OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runs/tai8_v1}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}
MAX_STEPS=${MAX_STEPS:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
VAL_INTERVAL=${VAL_INTERVAL:-2000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
WARMUP_STEPS=${WARMUP_STEPS:-500}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUTPUT_DIR"

# One --train-manifest / --val-manifest per shard. Shards are speaker-disjoint,
# so the union is exactly the full corpus with no duplicated speakers.
manifest_args=()
for ((i = 0; i < NUM_GPUS; i++)); do
    train_pairs="$WORK/processed/shard${i}/train/gpt_pairs.jsonl"
    val_pairs="$WORK/processed/shard${i}/val/gpt_pairs.jsonl"
    [[ -f "$train_pairs" ]] && manifest_args+=(--train-manifest "${train_pairs}::zh")
    [[ -f "$val_pairs" ]] && manifest_args+=(--val-manifest "${val_pairs}::zh")
done

if [[ ${#manifest_args[@]} -eq 0 ]]; then
    echo "[Error] no pair manifests found under $WORK/processed — run 03_build_pairs.sh first" >&2
    exit 1
fi

echo "[Info] NUM_GPUS=$NUM_GPUS batch=$BATCH_SIZE accum=$GRAD_ACCUM"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    trainers/train_gpt_v2.py \
    "${manifest_args[@]}" \
    --tokenizer "$CKPT/bpe.model" \
    --config "$CKPT/config.yaml" \
    --base-checkpoint "$CKPT/gpt.pth" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accumulation "$GRAD_ACCUM" \
    --epochs "$EPOCHS" \
    --learning-rate "$LR" \
    --max-steps "$MAX_STEPS" \
    --num-workers "$NUM_WORKERS" \
    --warmup-steps "$WARMUP_STEPS" \
    --log-interval "$LOG_INTERVAL" \
    --val-interval "$VAL_INTERVAL" \
    --grad-clip 1.0 \
    --text-loss-weight 0.2 \
    --mel-loss-weight 0.8 \
    --amp \
    --resume auto \
    $EXTRA_ARGS
