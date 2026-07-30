#!/usr/bin/env bash
# Stage 3 (CPU only): build prompt/target pairs for every shard.
#
# The pair manifest is written *beside* the per-shard feature manifest, because
# the trainer resolves relative feature paths against the manifest's own
# directory. Putting it anywhere else would break every path in the file.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO"

PAIRS_PER_TARGET=${PAIRS_PER_TARGET:-1}
MIN_PROMPT_DURATION=${MIN_PROMPT_DURATION:-2.5}

PROCESSED=$WORK/$CORPUS/processed

for ((i = 0; i < NUM_GPUS; i++)); do
    for split in train val; do
        src="$PROCESSED/shard${i}/${split}/train_manifest.jsonl"
        dst="$PROCESSED/shard${i}/${split}/gpt_pairs.jsonl"
        if [[ ! -f "$src" ]]; then
            echo "[Skip] missing $src"
            continue
        fi
        echo "=== ${CORPUS}/shard${i}/${split} ==="
        python tools/build_gpt_prompt_pairs.py \
            --manifest "$src" \
            --output "$dst" \
            --pairs-per-target "$PAIRS_PER_TARGET" \
            --min-prompt-duration "$MIN_PROMPT_DURATION"
    done
done

echo "=== pair counts ==="
find "$PROCESSED" -name 'gpt_pairs.jsonl' | sort | xargs wc -l
