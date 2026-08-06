#!/usr/bin/env bash
# Runs inside the container: one clustering process per GPU, each taking a
# stride of the episode list, then concatenates the per-shard manifests.
#
# Args: NUM_GPUS

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

NUM_GPUS=${1:-4}
OUT_DIR=$WORK/resegment/labelled
mkdir -p "$OUT_DIR" "$LOGS"
cd "$REPO"

CAMPPLUS=${CAMPPLUS:-$CKPT/campplus_cn_common.bin}
[[ -f "$CAMPPLUS" ]] || { echo "[Error] missing $CAMPPLUS" >&2; exit 1; }

nvidia-smi --query-gpu=index,name,memory.total --format=csv

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    log=$LOGS/assign_speakers_shard${i}.log
    echo "[Launch] shard$i -> GPU $i  (log: $log)"
    (
        export CUDA_VISIBLE_DEVICES=$i
        CORPUS_ROOT=/mnt/shared/p06/taigi_resegment/segments \
        TAI8_MANIFESTS=$DATASET/tai8/manifests/index_tts \
        CAMPPLUS=$CAMPPLUS \
        OUT=$OUT_DIR/part${i}.jsonl \
        STATS=$OUT_DIR/part${i}_stats.json \
        DISTANCE=${DISTANCE:-0.50} \
        WORKERS=${WORKERS:-12} \
        SHARD=$i NUM_SHARDS=$NUM_GPUS \
            python tools/assign_speakers_by_episode.py
    ) >"$log" 2>&1 &
    pids+=($!)
done

status=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then echo "[Done] shard$idx"
    else echo "[FAILED] shard$idx — see $LOGS/assign_speakers_shard${idx}.log" >&2; status=1; fi
done

cat "$OUT_DIR"/part*.jsonl >"$OUT_DIR/all.jsonl"
echo "=== total ==="
wc -l "$OUT_DIR"/part*.jsonl "$OUT_DIR/all.jsonl"
python - "$OUT_DIR/all.jsonl" <<'PY'
import json, sys, collections
spk = collections.Counter(); dur = 0.0; n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(line); spk[d["speaker"]] += 1; dur += d["duration"]; n += 1
sizes = sorted(spk.values())
print(f"  clips {n:,}   hours {dur/3600:.1f}   pseudo-speakers {len(spk):,}")
print(f"  clips per speaker: median {sizes[len(sizes)//2]}  max {sizes[-1]}")
PY
exit "$status"
