#!/usr/bin/env bash
#SBATCH --job-name=itts15-extract
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/itts15_extract_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/itts15_extract_%j.out
#
# Feature extraction for the IndexTTS 1.5 comparison run, one process per GPU.
#
# --extract_condition is deliberately omitted. It exists to find a medoid
# conditioning vector per speaker, which only the learnable-speaker-embedding
# path consumes, and that path is off: every call site passes speaker_ids=None
# and the trainer derives its conditioning from a sampled clip's mel instead.
# Running it would also write a full pairwise distance matrix per part — at
# ~70k clips each that is about 20 GB of a file nothing reads.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/13_extract_15.sbatch

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${NUM_GPUS:-4}

srun --ntasks=1 \
    --container-image /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT15/repo/deploy/taipei1/_extract_15_inner.sh" "$NUM_GPUS"
