#!/usr/bin/env bash
#SBATCH --job-name=itts15-variants
#SBATCH --partition=p06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/shared/p06/indextts2/logs/itts15_variants_%j.out
#SBATCH --error=/mnt/shared/p06/indextts2/logs/itts15_variants_%j.out
#
# Two LoRA variants, back to back, after the first finetune lost the base
# model's hold on timbre across sentence joins.
#
# The stock Chinese 1.5 has no seam problem, so the binding is there to start
# with and finetuning is what loosened it. Conditioning arrives as 32 prefix
# latents the model reads through attention, which is where that binding lives;
# mapping Mandarin orthography to Taiwanese pronunciation is closer to lexical
# knowledge, which tends to live in the MLPs. Restricting LoRA to the MLPs
# should then keep the binding and still learn the pronunciation, rather than
# trading one against the other the way a smaller learning rate would.
#
#   mlp_only  LoRA on mlp.c_fc / mlp.c_proj only, lr unchanged at 5e-5
#   low_lr    all four modules as before, lr 1e-5
#
# low_lr is the control: if mlp_only works it says whether avoiding attention is
# what did it, or merely a smaller update.
#
#   NUM_GPUS=4 sbatch --gres=gpu:h100:4 deploy/taipei1/15_lora_variants.sbatch

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${NUM_GPUS:-4}

srun --ntasks=1 \
    --container-image /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
    --container-mounts /mnt/shared/p06:/mnt/shared/p06 \
    bash "$ROOT15/repo/deploy/taipei1/_lora_variants_inner.sh" "$NUM_GPUS"
