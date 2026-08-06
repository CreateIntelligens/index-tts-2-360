#!/usr/bin/env bash
#SBATCH --job-name=itts-spkassign
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/spkassign_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/spkassign_%j.out
#
# Label the resegment corpus so it can be paired.
#
# It carries ~390 h over 1,092 episodes with transcripts but no speakers, and
# pairing needs "same speaker" to hold. Clustering inside each episode supplies
# that: pairing never needs to know *which* person, only that two clips are the
# same one, and confining clusters to an episode keeps the problem small.
#
# Reads only. Writes manifests under $ROOT/work/resegment/.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/12_assign_speakers.sbatch

set -euo pipefail

ROOT=/mnt/shared/p06/indextts2
NUM_GPUS=${NUM_GPUS:-4}

srun --ntasks=1 \
    --container-image "$ROOT/images/indextts2-vllm.sqsh" \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT/repo/deploy/taipei1/_assign_speakers_inner.sh" "$NUM_GPUS"
