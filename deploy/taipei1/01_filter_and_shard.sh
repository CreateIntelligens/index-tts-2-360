#!/usr/bin/env bash
# Stage 1 (CPU only): drop untokenisable lines, then split by speaker into
# NUM_GPUS shards ready for parallel preprocessing.
#
# CORPUS selects which corpus to prepare and keeps each one's filtered manifests
# and shards in their own directory, so preparing a second corpus never clobbers
# the first. SRC_MANIFESTS overrides the source directory for anything not listed
# in env.sh.
#
#   CORPUS=tai8 bash .../01_filter_and_shard.sh
#   CORPUS=naer bash .../01_filter_and_shard.sh
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

FILTERED=$WORK/$CORPUS/filtered
SHARDS=$WORK/$CORPUS/shards
mkdir -p "$FILTERED"

echo "=== corpus: $CORPUS  source: $SRC_MANIFESTS ==="
for split in train val; do
    echo "=== filtering ${split} ==="
    python tools/filter_manifest.py \
        --manifest "$SRC_MANIFESTS/${split}_manifest.jsonl" \
        --output "$FILTERED/${split}_manifest.jsonl" \
        --tokenizer "$CKPT/bpe.model" \
        --language zh \
        --zh-to-simplified \
        --report "$FILTERED/${split}_filter_report.json"
done

echo "=== sharding by speaker into ${NUM_GPUS} shards ==="
python tools/shard_manifest.py \
    --train-manifest "$FILTERED/train_manifest.jsonl" \
    --val-manifest "$FILTERED/val_manifest.jsonl" \
    --output-dir "$SHARDS" \
    --shards "$NUM_GPUS"

echo "=== done ==="
find "$SHARDS" -name '*.jsonl' | sort | xargs wc -l
