#!/usr/bin/env bash
#SBATCH --job-name=itts15-extract-v5
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/itts15_extract_v5_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/itts15_extract_v5_%j.out
#
# Feature extraction for v5: tai8 + naer + the 850 episodes of
# segment_data/output that tai8 does not cover. 868k clips against the 282k the
# previous run used, so roughly 48 minutes on 4 GPUs by the 15m38s that one took.
#
# Writes to finetune_data_v5/ rather than finetune_data/, so the features the
# earlier runs were trained on stay exactly as they are and either can be
# reproduced.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/17_extract_v5.sbatch

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${NUM_GPUS:-4}

srun --ntasks=1 \
    --container-image /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT15/repo/deploy/taipei1/_extract_v5_inner.sh" "$NUM_GPUS"
