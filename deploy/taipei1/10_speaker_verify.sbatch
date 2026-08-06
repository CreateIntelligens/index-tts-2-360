#!/usr/bin/env bash
#SBATCH --job-name=itts-spkver
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/spkver_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/spkver_%j.out
#
# Gate before spending any of the remaining cluster time on the resegment corpus.
#
# That corpus has ~390 h of audio over 1,092 episodes but no speaker labels, and
# only 242 of those episodes were ever labelled by hand. Pairing needs "same
# speaker" to be true, so the labels have to come from somewhere; this measures
# whether CAMPPlus can supply them, using tai8's own labels as ground truth on
# episodes withheld from enrolment.
#
# Reads only. Writes one directory under $ROOT/analysis/.
#
#   sbatch --gres=gpu:h100:1 deploy/taipei1/10_speaker_verify.sbatch

set -euo pipefail

ROOT=/mnt/shared/p06/indextts2
OUT_DIR=${OUT_DIR:-$ROOT/analysis/speaker_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_DIR"

srun --ntasks=1 \
    --container-image "$ROOT/images/indextts2-vllm.sqsh" \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT/repo/deploy/taipei1/_speaker_verify_inner.sh" "$OUT_DIR"
