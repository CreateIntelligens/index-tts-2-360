#!/usr/bin/env bash
#SBATCH --job-name=itts15-v5
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/itts15_v5_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/itts15_v5_%j.out
#
# v5: full LoRA on tai8 + naer + output850.
#
# 15 epochs, every one kept. The corpus is 868k clips, so an epoch is 3,053
# steps per card and the whole run is about 42 hours on 4 GPUs. That is far past
# where the previous experiment stopped improving, and deliberately so: the
# checkpoint is chosen by listening, and the only way to hear where it turns is
# to have the ones on the far side of it.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/18_train_v5.sbatch

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-15}
NAME=${NAME:-v5_full_15ep}

srun --ntasks=1 \
    --container-image /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT15/repo/deploy/taipei1/_train_v5_inner.sh" "$NUM_GPUS" "$EPOCHS" "$NAME"
