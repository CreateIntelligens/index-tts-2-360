#!/usr/bin/env bash
#SBATCH --job-name=itts15-epochs
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/itts15_epochs_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/itts15_epochs_%j.out
#
# The same two LoRA settings again at 10 epochs instead of 4.
#
# Validation loss had all but flattened by the fourth pass, but it scores
# next-token prediction on held-out clips of the same drama corpus, which is not
# the same question as whether a Mandarin word comes out with the right
# Taiwanese reading. The two can part company: a run can keep getting the
# pronunciation right more often while the loss sits still or drifts up.
#
# A checkpoint is written every epoch, so one long run yields every intermediate
# point and the best can be chosen by ear rather than by loss.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/16_more_epochs.sbatch

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${NUM_GPUS:-4}

srun --ntasks=1 \
    --container-image /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT15/repo/deploy/taipei1/_more_epochs_inner.sh" "$NUM_GPUS"
