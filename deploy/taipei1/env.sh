#!/usr/bin/env bash
# Shared paths + container environment for the Taipei-1 runs.
# Sourced by the other scripts in this directory; also safe to source by hand.

set -euo pipefail

export ROOT=/mnt/shared/p06/indextts2
export REPO=$ROOT/repo
export CKPT=$ROOT/checkpoints
export WORK=$ROOT/work
export LOGS=$ROOT/logs
export IMAGE=$ROOT/images/indextts2-vllm.sqsh
export DATASET=/mnt/shared/p06/dataset202607_1

# Which corpus the stage scripts operate on. Each corpus keeps its filtered
# manifests, shards and features under $WORK/$CORPUS/ so preparing a second one
# cannot overwrite the first.
export CORPUS=${CORPUS:-tai8}
export SRC_MANIFESTS=${SRC_MANIFESTS:-$DATASET/$CORPUS/manifests/index_tts}
export TAI8=$DATASET/tai8/manifests/index_tts   # kept for older invocations

# Packages the image predates (e.g. opencc). The container filesystem is
# read-only, so extra deps live on the shared mount and are prepended here.
export PYLIBS=$ROOT/pylibs

# PYTHONPATH must win over the copy of indextts baked into the image at /app,
# otherwise the fixes in this repo (semantic-code trimming, the zh normaliser
# switch) silently would not apply.
export PYTHONPATH=$REPO:$PYLIBS
export HF_HOME=$CKPT/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HF_XET_CACHE=$HF_HOME/xet
export HF_HUB_DISABLE_XET=1
export INDEXTTS_USE_DEEPSPEED=0
# Traditional-Chinese corpus: keep training and inference on the same text path.
export INDEXTTS_ZH_T2S=1

export SLURM_PARTITION=${SLURM_PARTITION:-p06}
# How many GPUs to use. Nothing downstream hard-codes this.
export NUM_GPUS=${NUM_GPUS:-4}

CONTAINER_MOUNTS="/mnt/shared/p06:/mnt/shared/p06"
export CONTAINER_MOUNTS
