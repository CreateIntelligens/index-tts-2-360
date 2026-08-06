#!/usr/bin/env bash
# Runs inside the container: build the held-out-episode split from tai8's own
# speaker labels, then score CAMPPlus assignment on it.
#
# Args: OUT_DIR

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUT_DIR=${1:?usage: _speaker_verify_inner.sh OUT_DIR}
mkdir -p "$OUT_DIR"
cd "$REPO"

# The cluster has no route to huggingface.co, so the weights are staged on the
# shared mount rather than fetched by hf_hub_download the way inference does it.
CAMPPLUS=${CAMPPLUS:-$CKPT/campplus_cn_common.bin}
if [[ ! -f "$CAMPPLUS" ]]; then
    echo "[Error] missing CAMPPlus weights at $CAMPPLUS" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo "=== split ==="
TAI8_MANIFESTS=$DATASET/tai8/manifests/index_tts \
OUT=$OUT_DIR/speaker_split.json \
    python tools/build_speaker_split.py

echo
echo "=== verify ==="
SPLIT=$OUT_DIR/speaker_split.json \
AUDIO_ROOT=$DATASET/tai8/manifests/index_tts \
CAMPPLUS=$CAMPPLUS \
OUT=$OUT_DIR/speaker_verify.npz \
WORKERS=${WORKERS:-32} \
    python tools/speaker_verify.py

echo
echo "results in $OUT_DIR"
