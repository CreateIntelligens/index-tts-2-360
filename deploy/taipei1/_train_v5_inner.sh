#!/usr/bin/env bash
# Runs inside the container: v5, full LoRA, on tai8 + naer + output850.
#
# Two settings carry the findings from the runs before this one.
#
# Full LoRA rather than mlp_only. The mlp_only variant existed to test whether
# leaving attention alone would keep the base model's timbre binding and stop
# the voice shifting at sentence joins. Listening said full LoRA sounds better,
# and scoring the two through Breeze-ASR-26 agreed: 0.135 against 0.197 mean CER
# over the same sentences and seeds.
#
# 4,000 steps per card, not 4 passes over the data. The 10-epoch run showed that
# training past this point starts dropping words while validation loss keeps
# falling — loss is computed under teacher forcing and never sees free-running
# generation, so it cannot register the failure. 4,000 is what the best
# checkpoint so far was trained for; the corpus is 3.1x larger, so the same
# number of updates now covers three times the vocabulary. That is the point of
# this run, since the errors are word-coverage errors.
#
# Every epoch is kept and early stopping is off: the choice between them is made
# with tools/asr_eval.py and ears, never with the loss.
#
# Args: NUM_GPUS EPOCHS NAME

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
NUM_GPUS=${1:-4}
EPOCHS=${2:-15}
NAME=${3:-v5_full_15ep}
export PYTHONPATH=$ROOT15/repo:$ROOT15/pylibs
export OMP_NUM_THREADS=8
cd "$ROOT15/repo"

RUNS=$ROOT15/repo/finetune_models/runs
mkdir -p "$RUNS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "============================================================"
echo "  v5: $NAME  ($EPOCHS epochs, every one kept)"
echo "============================================================"

python - "$EPOCHS" <<'PY'
import sys
from omegaconf import OmegaConf

epochs = int(sys.argv[1])
cfg = OmegaConf.load("finetune_models/config_finetuned.yaml")

cfg.train.epochs = epochs
cfg.train.early_stopping_patience = epochs + 1
cfg.train.num_workers = 0
cfg.train.valid_num_workers = 0
cfg.train.prefetch_factor = None
cfg.train.persistent_workers = False
cfg.train.lazy_load_metadata = True
# The v5 features, kept apart from the ones the earlier runs used.
cfg.train.data_path = "finetune_data_v5/processed_data"
cfg.gpt_checkpoint = "gpt.pth"
cfg.dvae_checkpoint = "dvae.pth"
cfg.bigvgan_checkpoint = "bigvgan_generator.pth"
# The four modules config_finetuned.yaml already names; spelled out so a change
# upstream cannot silently alter what this run means.
cfg.train.lora.target_modules = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

OmegaConf.save(cfg, "finetune_models/config.yaml")
print(f"  epochs={cfg.train.epochs}  "
      f"lr={cfg.train.optimizer.learning_rate}  "
      f"targets={list(cfg.train.lora.target_modules)}  "
      f"data={cfg.train.data_path}")
PY

rm -rf finetune_models/checkpoints
mkdir -p finetune_models/checkpoints
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train_ddp.py

mkdir -p "$RUNS/$NAME"
mv finetune_models/checkpoints/* "$RUNS/$NAME/" 2>/dev/null || true
cp finetune_models/config.yaml "$RUNS/$NAME/config.yaml"
# 2.3 GB each and only useful for resuming.
rm -f "$RUNS/$NAME"/checkpoint_epoch_*.pt
echo "[Done] $NAME"
ls -1 "$RUNS/$NAME"
