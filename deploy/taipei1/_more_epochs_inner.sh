#!/usr/bin/env bash
# Runs inside the container: both LoRA settings at 10 epochs, one after the
# other, each into its own output directory.
#
# Args: NUM_GPUS

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${1:-4}
EPOCHS=${EPOCHS:-10}
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
    echo "  variant: $name  (epochs=$EPOCHS)"
    echo "============================================================"

    python - "$name" "$EPOCHS" <<'PY'
import sys
from omegaconf import OmegaConf

name, epochs = sys.argv[1], int(sys.argv[2])
cfg = OmegaConf.load("finetune_models/config_finetuned.yaml")

cfg.train.epochs = epochs
cfg.train.max_steps_per_epoch = 1200
# Stopping on validation loss would defeat the point: the run is here to find
# out whether pronunciation keeps improving after the loss stops moving.
cfg.train.early_stopping_patience = epochs + 1
cfg.train.num_workers = 0
cfg.train.valid_num_workers = 0
cfg.train.prefetch_factor = None
cfg.train.persistent_workers = False
cfg.train.lazy_load_metadata = True
cfg.train.data_path = "finetune_data/processed_data"
cfg.gpt_checkpoint = "gpt.pth"
cfg.dvae_checkpoint = "dvae.pth"
cfg.bigvgan_checkpoint = "bigvgan_generator.pth"

if name == "mlp_only_10ep":
    cfg.train.lora.target_modules = ["mlp.c_fc", "mlp.c_proj"]
elif name == "full_10ep":
    pass   # the four modules config_finetuned.yaml already names
else:
    raise SystemExit(f"unknown variant {name}")

OmegaConf.save(cfg, "finetune_models/config.yaml")
print(f"  epochs={cfg.train.epochs}  lr={cfg.train.optimizer.learning_rate}  "
      f"targets={list(cfg.train.lora.target_modules)}  "
      f"early_stop={cfg.train.early_stopping_patience}")
PY

    rm -rf finetune_models/checkpoints
    mkdir -p finetune_models/checkpoints
    torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train_ddp.py

    mkdir -p "$RUNS/$name"
    mv finetune_models/checkpoints/* "$RUNS/$name/" 2>/dev/null || true
    cp finetune_models/config.yaml "$RUNS/$name/config.yaml"
    # The optimiser states are 2.3 GB each and only matter for resuming.
    rm -f "$RUNS/$name"/checkpoint_epoch_*.pt
    echo "[Done] $name"
    ls -1 "$RUNS/$name" | tail -12
}

for v in mlp_only_10ep full_10ep; do
    run_variant "$v"
done

echo ""
echo "=== all runs ==="
find "$RUNS" -name 'gpt_epoch_*.pth' -printf '  %p\n' | sort
