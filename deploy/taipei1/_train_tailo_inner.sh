#!/usr/bin/env bash
# Continued training on SuíSiann, to find out whether IndexTTS can be taught to
# read Tâi-lô annotation the way CosyVoice3 was.
#
# Starts from v5_full_15ep epoch 8, the checkpoint the 6,000-take ASR comparison
# put first among the Taiwanese models, rather than from the base weights: the
# point is to add a capability without giving up the Taiwanese already there.
#
# Trained on SuíSiann alone, not mixed into the 668 hours. 4.73 hours diluted
# into that is 0.7% of the corpus and would very likely not be learned at all;
# two stages is the shape that has actually been made to work.
#
# One speaker throughout (王秀容). That was expected to cause catastrophic
# forgetting on the Han side and did not, in the one run this has been tried in.
# It is the main thing to listen for.
#
# Extraction runs on a single GPU: the conditioning sampler draws its prompt
# from another clip by the same speaker in the same part, and with one speaker
# every clip lands in part 0 regardless of how many parts are asked for.
#
# Args: NUM_GPUS EPOCHS NAME

set -euo pipefail

ROOT15=/mnt/shared/p06/indextts15
DATA=$ROOT15/finetune_data_tailo
SRC=/mnt/shared/p06/suisiann_tailo
LOGS=/mnt/shared/p06/indextts2/logs
NUM_GPUS=${1:-4}
EPOCHS=${2:-8}
NAME=${3:-tailo_v1}
# Relative to the repo, which is the working directory — and the runs tree
# lives under finetune_models, not at the top. Job 179703 died here after
# extraction had already succeeded.
BASE=${BASE:-finetune_models/runs/v5_full_15ep/gpt_epoch_8.pth}
export PYTHONPATH=$ROOT15/repo:$ROOT15/pylibs
export OMP_NUM_THREADS=8
cd "$ROOT15/repo"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p "$DATA/audio_list" "$DATA/processed_data"
# train_ddp.py resolves data_path relative to the repo; extraction writes outside it.
ln -sfn "$DATA" "$ROOT15/repo/finetune_data_tailo"

echo "=== audio_list ==="
cp "$SRC/audio_list.txt" "$DATA/audio_list/audio_list_part_0.txt"
wc -l "$DATA/audio_list/audio_list_part_0.txt"
# head reads the file directly. Piping cut into head kills the job under
# `set -o pipefail`: head exits after one line, cut takes SIGPIPE, and the
# pipeline's non-zero status ends the script — a diagnostic line that printed
# one path was enough to fail job 179691 after 14 seconds.
ls -la "$(head -1 "$DATA/audio_list/audio_list_part_0.txt" | cut -f1)"

if [ ! -f "$DATA/processed_data/speaker_info.json" ]; then
    echo "=== 抽取特徵 ==="
    CUDA_VISIBLE_DEVICES=0 python tools/extract_codec.py \
        --audio_list "$DATA/audio_list/audio_list_part_0.txt" \
        --output_dir "$DATA/processed_data/" \
        --device cuda --batch_size 16 --num_workers 8 \
        >"$LOGS/itts15_tailo_extract.log" 2>&1
    AUDIO_LIST_DIR=$DATA/audio_list PROCESSED_DIR=$DATA/processed_data \
        python tools/backfill_speaker_ids.py
    python tools/generate_speaker_info.py "$DATA/processed_data" 2>&1 | tail -3
else
    echo "=== 特徵已存在，跳過抽取 ==="
fi

echo "=== 設定 ==="
# A checkpoint written by training carries one key the fresh model does not
# have — gpt.wte.weight, 512 against the base's 511 — and load_UnifiedVoice
# loads with strict=True, so a merged checkpoint cannot be used as a starting
# point unmodified. Job 179704 died here. Nothing else differs, so dropping the
# key is enough; the file is rewritten rather than the loader relaxed, because
# strict=True is what catches a genuinely wrong checkpoint.
python - "$BASE" <<'PYEOF'
import sys, torch
src = sys.argv[1]
state = torch.load(src, map_location="cpu")
inner = state.get("model", state)
removed = inner.pop("gpt.wte.weight", None)
torch.save(state, "finetune_models/gpt_tailo_base.pth")
print(f"  起點 {src}")
print(f"  移除 gpt.wte.weight: {'有' if removed is not None else '本來就沒有'}  剩 {len(inner)} 個鍵")
PYEOF
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
cfg.train.data_path = "finetune_data_tailo/processed_data"
# Continue from the best Taiwanese checkpoint, not the base weights.
cfg.gpt_checkpoint = "gpt_tailo_base.pth"
cfg.dvae_checkpoint = "dvae.pth"
cfg.bigvgan_checkpoint = "bigvgan_generator.pth"
# 1e-5 rather than the 5e-5 the corpus runs used. This is a small, single
# speaker set and the Han ability already in the weights is what is at risk.
cfg.train.optimizer.learning_rate = 1e-5
cfg.train.lora.target_modules = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
OmegaConf.save(cfg, "finetune_models/config.yaml")
print(f"  epochs={cfg.train.epochs}  lr={cfg.train.optimizer.learning_rate}  "
      f"warmup_ratio={cfg.train.optimizer.warmup_ratio}  "
      f"base={cfg.gpt_checkpoint}  data={cfg.train.data_path}")
PY

rm -rf finetune_models/checkpoints
mkdir -p finetune_models/checkpoints
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train_ddp.py

RUNS=$ROOT15/repo/finetune_models/runs
mkdir -p "$RUNS/$NAME"
mv finetune_models/checkpoints/* "$RUNS/$NAME/" 2>/dev/null || true
cp finetune_models/config.yaml "$RUNS/$NAME/config.yaml"
rm -f "$RUNS/$NAME"/checkpoint_epoch_*.pt
echo "[Done] $NAME"
ls -1 "$RUNS/$NAME"
