#!/usr/bin/env bash
#SBATCH --job-name=itts-spkclust
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/spkclust_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/spkclust_%j.out
#
# Second gate on the resegment corpus, after enrolment against tai8's identities
# failed (63% top-1; holding 98% precision left 3% of clips).
#
# Clustering within one episode is a smaller problem than open-set matching
# across 147 identities, and it is all that pairing needs: a pair must be the
# same speaker, but need not be a *named* speaker. Scored on tai8's labelled
# episodes, by the pairwise precision of the pairs it would produce.
#
# Reads only. Writes one directory under $ROOT/analysis/.
#
#   sbatch --gres=gpu:h100:1 deploy/taipei1/11_speaker_cluster.sbatch

set -euo pipefail

ROOT=/mnt/shared/p06/indextts2
OUT_DIR=${OUT_DIR:-$ROOT/analysis/cluster_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_DIR"

srun --ntasks=1 \
    --container-image "$ROOT/images/indextts2-vllm.sqsh" \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT/repo/deploy/taipei1/_speaker_cluster_inner.sh" "$OUT_DIR"
