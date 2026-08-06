#!/usr/bin/env bash
# Runs inside the container: one extract_codec per GPU, each on its own part,
# then backfills the speaker each clip belongs to and writes speaker_info.json.
#
# Args: NUM_GPUS

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
LOGS=/mnt/shared/p06/indextts2/logs
NUM_GPUS=${1:-4}
export PYTHONPATH=$ROOT15/repo:$ROOT15/pylibs
cd "$ROOT15/repo"

nvidia-smi --query-gpu=index,name,memory.total --format=csv

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    log=$LOGS/itts15_extract_part${i}.log
    echo "[Launch] part$i -> GPU $i  (log: $log)"
    (
        export CUDA_VISIBLE_DEVICES=$i
        python tools/extract_codec.py \
            --audio_list "$ROOT15/finetune_data/audio_list/audio_list_part_${i}.txt" \
            --output_dir "$ROOT15/finetune_data/processed_data/" \
            --device cuda --batch_size 16 --num_workers 8
    ) >"$log" 2>&1 &
    pids+=($!)
done

status=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then echo "[Done] part$idx"
    else echo "[FAILED] part$idx — see $LOGS/itts15_extract_part${idx}.log" >&2; status=1; fi
done
[[ $status -eq 0 ]] || exit "$status"

echo "=== backfilling speaker_id ==="
AUDIO_LIST_DIR=$ROOT15/finetune_data/audio_list \
PROCESSED_DIR=$ROOT15/finetune_data/processed_data \
    python tools/backfill_speaker_ids.py

echo "=== speaker_info.json ==="
python tools/generate_speaker_info.py "$ROOT15/finetune_data/processed_data" 2>&1 | tail -5
