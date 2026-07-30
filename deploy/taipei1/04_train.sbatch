#!/usr/bin/env bash
#SBATCH --job-name=itts-train
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/train_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/train_%j.out
#
# Stage 4: DDP finetuning. One torchrun process per GPU on a single node.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/04_train.sbatch
#
# The GPU count is not baked in anywhere: change --gres and NUM_GPUS together.
# Submitted via sbatch so the run outlives the SSH session and the 48h cert.

set -euo pipefail

ROOT=/mnt/shared/p06/indextts2

srun --ntasks=1 \
    --container-image "$ROOT/images/indextts2-vllm.sqsh" \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT/repo/deploy/taipei1/_train_inner.sh"
