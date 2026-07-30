#!/usr/bin/env bash
# Stage 1 (CPU only): drop untokenisable lines, then split by speaker into
# NUM_GPUS shards ready for parallel preprocessing.
#
# Run inside the container:
#   srun --ntasks=1 -p p06 --container-image ... --container-mounts ... \
#        bash /mnt/shared/p06/indextts2/repo/deploy/taipei1/01_filter_and_shard.sh

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO"

echo "=== sanity: which indextts is on the path? ==="
python - <<'PY'
import indextts
import indextts.utils.front as front
print("indextts   :", indextts.__file__)
print("front      :", front.__file__)
print("has t2s map:", hasattr(front.TextNormalizer, "ZH_VARIANT_MAP"))
assert hasattr(front.TextNormalizer, "ZH_VARIANT_MAP"), \
    "The image's bundled indextts is shadowing the repo; PYTHONPATH is wrong."
import tools.preprocess_data  # noqa: F401  (import smoke test)
print("preprocess_data imports OK")
PY

mkdir -p "$WORK/filtered"

for split in train val; do
    echo "=== filtering ${split} ==="
    python tools/filter_manifest.py \
        --manifest "$TAI8/${split}_manifest.jsonl" \
        --output "$WORK/filtered/${split}_manifest.jsonl" \
        --tokenizer "$CKPT/bpe.model" \
        --language zh \
        --zh-to-simplified \
        --report "$WORK/filtered/${split}_filter_report.json"
done

echo "=== sharding by speaker into ${NUM_GPUS} shards ==="
python tools/shard_manifest.py \
    --train-manifest "$WORK/filtered/train_manifest.jsonl" \
    --val-manifest "$WORK/filtered/val_manifest.jsonl" \
    --output-dir "$WORK/shards" \
    --shards "$NUM_GPUS"

echo "=== done ==="
find "$WORK/shards" -name '*.jsonl' | sort | xargs wc -l
