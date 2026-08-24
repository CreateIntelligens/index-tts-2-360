#!/usr/bin/env bash
#SBATCH --job-name=itts15-tailo
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/itts15_tailo_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/itts15_tailo_%j.out
#
# Teach the Tâi-lô annotation, continuing from v5 epoch 8 on SuíSiann's 4.73
# hours. Extraction and training in one job: 10,284 rows extract in a couple of
# minutes, and splitting it in two would mean queueing twice.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/19_train_tailo.sbatch

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-8}
NAME=${NAME:-tailo_v1}

srun --ntasks=1 \
    --container-image /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT15/repo/deploy/taipei1/_train_tailo_inner.sh" "$NUM_GPUS" "$EPOCHS" "$NAME"
