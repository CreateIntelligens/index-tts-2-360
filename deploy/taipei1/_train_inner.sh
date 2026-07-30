#!/usr/bin/env bash
# Runs inside the container: launches torchrun with one process per GPU.
#
# The GPU count comes from NUM_GPUS at call time, so the same script covers a
# 1-GPU smoke test and an N-GPU production run.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO"

# Each run gets its own timestamped directory so versions never overwrite each
# other. Pass OUTPUT_DIR explicitly to resume an existing run (--resume auto
# looks for latest.pth *inside* OUTPUT_DIR, so a fresh timestamp starts over).
RUN_TAG=${RUN_TAG:-tai8}
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR=$ROOT/runs/$(date +%Y%m%d_%H%M%S)__${RUN_TAG}
fi
# Exported so run_config.json records the values actually used, including the
# defaults that were never passed in.
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export EPOCHS=${EPOCHS:-2}
export LR=${LR:-1e-5}
export MAX_STEPS=${MAX_STEPS:-0}
export NUM_WORKERS=${NUM_WORKERS:-4}
export VAL_INTERVAL=${VAL_INTERVAL:-2000}
export LOG_INTERVAL=${LOG_INTERVAL:-20}
export WARMUP_STEPS=${WARMUP_STEPS:-500}
export EXTRA_ARGS=${EXTRA_ARGS:-}
export RUN_TAG OUTPUT_DIR CORPORA

mkdir -p "$OUTPUT_DIR"

# One --train-manifest / --val-manifest per shard, for every corpus in CORPORA.
# Shards are speaker-disjoint, so the union is the whole corpus with no duplicated
# speakers. Legacy runs kept tai8 directly under $WORK, so that layout is accepted
# as a fallback.
CORPORA=${CORPORA:-$CORPUS}
manifest_args=()
for corpus in $CORPORA; do
    base=$WORK/$corpus/processed
    [[ -d "$base" ]] || base=$WORK/processed          # pre-CORPUS layout
    found=0
    for ((i = 0; i < NUM_GPUS; i++)); do
        train_pairs="$base/shard${i}/train/gpt_pairs.jsonl"
        val_pairs="$base/shard${i}/val/gpt_pairs.jsonl"
        [[ -f "$train_pairs" ]] && { manifest_args+=(--train-manifest "${train_pairs}::zh"); found=1; }
        [[ -f "$val_pairs" ]] && manifest_args+=(--val-manifest "${val_pairs}::zh")
    done
    if [[ $found -eq 0 ]]; then
        echo "[Error] no pair manifests for corpus '$corpus' under $base — run 03_build_pairs.sh first" >&2
        exit 1
    fi
    echo "[Info] corpus '$corpus' -> $base"
done

if [[ ${#manifest_args[@]} -eq 0 ]]; then
    echo "[Error] no pair manifests found at all" >&2
    exit 1
fi

echo "[Info] NUM_GPUS=$NUM_GPUS batch=$BATCH_SIZE accum=$GRAD_ACCUM"
echo "[Info] OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Record what produced this run, so a directory is self-describing months later.
python - "$OUTPUT_DIR" "$RUN_TAG" <<'PY'
import json
import os
import sys
from datetime import datetime

output_dir, run_tag = sys.argv[1], sys.argv[2]
config = {
    "run_tag": run_tag,
    "started": datetime.now().isoformat(timespec="seconds"),
    "repo_commit": os.environ.get("GIT_COMMIT", "unknown"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "num_gpus": int(os.environ.get("NUM_GPUS", "1")),
    "hyperparams": {
        key: os.environ.get(key)
        for key in (
            "BATCH_SIZE", "GRAD_ACCUM", "EPOCHS", "LR", "MAX_STEPS",
            "WARMUP_STEPS", "VAL_INTERVAL", "NUM_WORKERS", "EXTRA_ARGS",
        )
    },
    "text_path": {
        "INDEXTTS_ZH_T2S": os.environ.get("INDEXTTS_ZH_T2S"),
    },
    "corpora": os.environ.get("CORPORA", os.environ.get("CORPUS", "tai8")),
    "data": os.environ.get("DATA_NOTE", "dataset202607_1, speaker-sharded"),
}
path = os.path.join(output_dir, "run_config.json")
os.makedirs(output_dir, exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
print(f"[Info] wrote {path}")
PY

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
