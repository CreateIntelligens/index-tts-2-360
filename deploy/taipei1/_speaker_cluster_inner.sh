#!/usr/bin/env bash
# Runs inside the container: cluster each labelled tai8 episode and score the
# result by the pairwise precision of the pairs it would allow.
#
# Args: OUT_DIR

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUT_DIR=${1:?usage: _speaker_cluster_inner.sh OUT_DIR}
mkdir -p "$OUT_DIR"
cd "$REPO"

CAMPPLUS=${CAMPPLUS:-$CKPT/campplus_cn_common.bin}
if [[ ! -f "$CAMPPLUS" ]]; then
    echo "[Error] missing CAMPPlus weights at $CAMPPLUS" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv

TAI8_MANIFESTS=$DATASET/tai8/manifests/index_tts \
CAMPPLUS=$CAMPPLUS \
OUT=$OUT_DIR/cluster_eval.npz \
EPISODES=${EPISODES:-40} \
MAX_CLIPS=${MAX_CLIPS:-400} \
WORKERS=${WORKERS:-32} \
    python tools/speaker_cluster_eval.py

echo
echo "results in $OUT_DIR"
