#!/usr/bin/env bash
# Shared paths + container environment for the Taipei-1 runs.
# Sourced by the other scripts in this directory; also safe to source by hand.

set -euo pipefail

# The Slurm clients live in /cm/local/apps/slurm/current/bin and are not on the
# default PATH; without this, sbatch/squeue/sinfo are simply "command not found"
# on the login node. Harmless inside the container, where there is no module
# command and no need for one.
if ! command -v sbatch >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load slurm 2>/dev/null || true
fi

export ROOT=/mnt/shared/p06/indextts2
export REPO=$ROOT/repo
export CKPT=$ROOT/checkpoints
export WORK=$ROOT/work
export LOGS=$ROOT/logs
export REFS=$ROOT/refs
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

# One directory per trained version, directly under $ROOT and named
# <version>_<YYYYmmdd_HHMMSS>, e.g. v3a_20260730_154608. Everything about that
# version lives inside it, so no artefact is ever ambiguous about which model
# produced it:
#
#   v3a_20260730_154608/
#     run_config.json   what produced it (settings, commit, job id, corpora)
#     model_step*.pth   training checkpoints (optimiser state included)
#     latest.pth        resume point
#     logs/             TensorBoard events
#     train.log         the Slurm job log
#     pruned/gpt.pth    inference weights, optimiser state stripped
#     eval/             generated samples + index.tsv
#
# The version label comes first so the directories read in release order at a
# glance; the timestamp keeps same-day runs distinct.
export VERSION_GLOB='v*_2*'

list_runs_paths() {
    ls -d "$ROOT"/$VERSION_GLOB/ 2>/dev/null | sed 's:/*$::'
}

# Newest by timestamp, which is the trailing part of the name.
latest_run() {
    list_runs_paths | awk -F/ '{n=$NF; sub(/^v[^_]*_/, "", n); print n"\t"$0}' |
        sort | tail -1 | cut -f2-
}

# Accept a full path, a directory name, a unique substring, or "latest".
resolve_run() {
    local want=${1:-latest}
    if [[ "$want" == latest ]]; then
        latest_run
        return
    fi
    if [[ -d "$want" ]]; then
        echo "${want%/}"
        return
    fi
    if [[ -d "$ROOT/$want" ]]; then
        echo "$ROOT/$want"
        return
    fi
    local matches count
    matches=$(list_runs_paths | grep -- "$want" || true)
    count=$(printf '%s\n' "$matches" | grep -c . || true)
    if [[ "$count" -eq 1 ]]; then
        printf '%s\n' "$matches"
        return
    fi
    if [[ "$count" -eq 0 ]]; then
        echo "[Error] no version matching '$want' under $ROOT" >&2
    else
        echo "[Error] '$want' matches $count versions:" >&2
        printf '  %s\n' $matches >&2
    fi
    return 1
}

export SLURM_PARTITION=${SLURM_PARTITION:-p06}
# How many GPUs to use. Nothing downstream hard-codes this.
export NUM_GPUS=${NUM_GPUS:-4}

CONTAINER_MOUNTS="/mnt/shared/p06:/mnt/shared/p06"
export CONTAINER_MOUNTS
