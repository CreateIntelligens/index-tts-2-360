#!/usr/bin/env bash
# Runs inside the container: one variant after another, each with its own config
# and its own output directory so nothing overwrites the run before it.
#
# Args: NUM_GPUS

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${1:-4}
export PYTHONPATH=$ROOT15/repo:$ROOT15/pylibs
export OMP_NUM_THREADS=8
cd "$ROOT15/repo"

RUNS=$ROOT15/repo/finetune_models/runs
mkdir -p "$RUNS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

run_variant () {
    local name=$1
    echo ""
    echo "============================================================"
    echo "  variant: $name"
    echo "============================================================"

    python - "$name" <<'PY'
import shutil, sys
from omegaconf import OmegaConf

name = sys.argv[1]
cfg = OmegaConf.load("finetune_models/config_finetuned.yaml")

# Same as the first run except where noted, so the comparison stays clean.
cfg.train.epochs = 4
cfg.train.max_steps_per_epoch = 1200
cfg.train.early_stopping_patience = 3
cfg.train.num_workers = 0
cfg.train.valid_num_workers = 0
cfg.train.prefetch_factor = None
cfg.train.persistent_workers = False
cfg.train.lazy_load_metadata = True
cfg.train.data_path = "finetune_data/processed_data"
cfg.gpt_checkpoint = "gpt.pth"
cfg.dvae_checkpoint = "dvae.pth"
cfg.bigvgan_checkpoint = "bigvgan_generator.pth"

if name == "mlp_only":
    # Leave attention alone: that is where the conditioning prefix is read.
    cfg.train.lora.target_modules = ["mlp.c_fc", "mlp.c_proj"]
elif name == "low_lr":
    cfg.train.optimizer.learning_rate = 1.0e-5
else:
    raise SystemExit(f"unknown variant {name}")

OmegaConf.save(cfg, "finetune_models/config.yaml")
print(f"  lr={cfg.train.optimizer.learning_rate}  "
      f"targets={list(cfg.train.lora.target_modules)}  "
      f"r={cfg.train.lora.r} alpha={cfg.train.lora.lora_alpha}")
PY

    rm -rf finetune_models/checkpoints
    mkdir -p finetune_models/checkpoints
    torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train_ddp.py

    mkdir -p "$RUNS/$name"
    mv finetune_models/checkpoints/* "$RUNS/$name/" 2>/dev/null || true
    cp finetune_models/config.yaml "$RUNS/$name/config.yaml"
    echo "[Done] $name -> $RUNS/$name"
    ls -lh "$RUNS/$name" | tail -4
}

for v in mlp_only low_lr; do
    run_variant "$v"
done

echo ""
echo "=== all variants ==="
find "$RUNS" -name 'gpt_epoch_4.pth' -printf '  %p  %s bytes\n'
