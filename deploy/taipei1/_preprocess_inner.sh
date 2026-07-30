#!/usr/bin/env bash
# Runs inside the container: launches one preprocess_data.py per GPU, each pinned
# to its own shard, then waits for all of them.
#
# Args: NUM_GPUS BATCH_SIZE BUCKET_SIZE WORKERS

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

NUM_GPUS=${1:-4}
BATCH_SIZE=${2:-16}
BUCKET_SIZE=${3:-256}
WORKERS=${4:-4}

cd "$REPO"
mkdir -p "$WORK/processed" "$LOGS"

echo "[Info] $NUM_GPUS shards, batch=$BATCH_SIZE bucket=$BUCKET_SIZE workers=$WORKERS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    shard_dir="$WORK/shards/shard${i}"
    out_dir="$WORK/processed/shard${i}"
    log="$LOGS/preprocess_shard${i}.log"

    if [[ ! -f "$shard_dir/train_manifest.jsonl" ]]; then
        echo "[Error] missing $shard_dir/train_manifest.jsonl" >&2
        exit 1
    fi

    echo "[Launch] shard$i -> GPU $i  (log: $log)"
    (
        # Each worker sees exactly one GPU, so the script's plain "cuda" is card i.
        export CUDA_VISIBLE_DEVICES=$i
        for split in train val; do
            python tools/preprocess_data.py \
                --manifest "$shard_dir/${split}_manifest.jsonl" \
                --output-dir "$out_dir/${split}" \
                --tokenizer "$CKPT/bpe.model" \
                --config "$CKPT/config.yaml" \
                --gpt-checkpoint "$CKPT/gpt.pth" \
                --language zh \
                --zh-to-simplified \
                --device cuda \
                --batch-size "$BATCH_SIZE" \
                --bucket-size "$BUCKET_SIZE" \
                --workers "$WORKERS" \
                --skip-existing \
                --val-ratio 0.0
        done
    ) >"$log" 2>&1 &
    pids+=($!)
done

status=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "[Done] shard$idx"
    else
        echo "[FAILED] shard$idx — see $LOGS/preprocess_shard${idx}.log" >&2
        status=1
    fi
done

echo "=== per-shard output ==="
for ((i = 0; i < NUM_GPUS; i++)); do
    for split in train val; do
        f="$WORK/processed/shard${i}/${split}/train_manifest.jsonl"
        [[ -f "$f" ]] && echo "  shard${i}/${split}: $(wc -l <"$f") records"
    done
done

exit "$status"
