#!/usr/bin/env bash
#SBATCH --job-name=itts15-train
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/itts15_train_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/itts15_train_%j.out
#
# IndexTTS 1.5 LoRA on the same corpora v4 used, as an architecture comparison.
#
# Hyperparameters come from finetune_models/config_finetuned.yaml, the config
# saved beside a completed run, rather than config.yaml at HEAD — that one holds
# the learnable-speaker-conditions experiment whose cloning broke. Only the step
# budget changed: their max_steps_per_epoch of 150 would show the model 3.4% of
# 282,300 clips per epoch.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/14_train_15.sbatch

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${NUM_GPUS:-4}

srun --ntasks=1 \
    --container-image /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash -c "
set -euo pipefail
cd $ROOT15/repo
export PYTHONPATH=$ROOT15/repo:$ROOT15/pylibs
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=index,name,memory.total --format=csv
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train_ddp.py
"
