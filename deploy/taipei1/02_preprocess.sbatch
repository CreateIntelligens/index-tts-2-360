#!/usr/bin/env bash
#SBATCH --job-name=itts-prep
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/prep_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/prep_%j.out
#
# Stage 2: feature extraction, one process per GPU, each on its own speaker shard.
#
# Submitted with sbatch (not srun --pty) so it survives the 48h SSH certificate
# expiry and any dropped connection.
#
#   sbatch --gres=gpu:h100:$NUM_GPUS deploy/taipei1/02_preprocess.sbatch
#
# Re-running is safe: --skip-existing means finished clips are not recomputed.

set -euo pipefail

ROOT=/mnt/shared/p06/indextts2
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
BUCKET_SIZE=${BUCKET_SIZE:-256}
WORKERS=${WORKERS:-4}

srun --ntasks=1 \
    --container-image "$ROOT/images/indextts2-vllm.sqsh" \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT/repo/deploy/taipei1/_preprocess_inner.sh" \
        "$NUM_GPUS" "$BATCH_SIZE" "$BUCKET_SIZE" "$WORKERS"
