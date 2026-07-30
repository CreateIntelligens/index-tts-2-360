#!/usr/bin/env bash
# Stage 5: synthesise with a trained version.
#
# RUN selects the version: a run directory, its name, a unique substring, or
# "latest" (the default). Everything derived from it stays inside that run
# directory, so artefacts never mix between versions.
#
#   RUN=latest                bash .../05_infer.sh
#   RUN=v3a                   bash .../05_infer.sh
#   RUN=20260730_104546__tai8_v2_emotarget  bash .../05_infer.sh
#
#   TEXT='你今天吃飽了嗎' PROMPT_WAV=/path/ref.wav RUN=v2 bash .../05_infer.sh
#
# TEXT is written in *Mandarin* Han characters, the same way the training
# transcripts were. The model maps Mandarin orthography to Taiwanese
# pronunciation, so there is no reason to write Taiwanese Hanji (今仔日, 食飽未)
# on the input side.
#
# INDEXTTS_ZH_T2S is exported by env.sh: inference must fold Traditional to
# Simplified exactly as preprocessing did, or the model sees text it never
# trained on.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO"

RUN_DIR=$(resolve_run "${RUN:-latest}") || exit 1
STEP=${STEP:-}   # e.g. STEP=11780 to pick a specific checkpoint
TEXT=${TEXT:-請你明天早上來這裡找我}
PROMPT_WAV=${PROMPT_WAV:-}

echo "=== run: $RUN_DIR ==="
[[ -f "$RUN_DIR/run_config.json" ]] && sed 's/^/    /' "$RUN_DIR/run_config.json"

# Pick the checkpoint: an explicit step, else the highest-numbered one.
if [[ -n "$STEP" ]]; then
    CKPT_SRC=$RUN_DIR/model_step${STEP}.pth
else
    CKPT_SRC=$(ls -1 "$RUN_DIR"/model_step*.pth 2>/dev/null |
        sed 's/.*model_step\([0-9]*\)\.pth/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [[ -z "$CKPT_SRC" || ! -f "$CKPT_SRC" ]]; then
    echo "[Error] no checkpoint in $RUN_DIR (looked for model_step*.pth)" >&2
    exit 1
fi

# Training checkpoints carry optimiser + scheduler state (~7.7GB). Strip it once
# per run and keep the result beside the version it came from.
PRUNED=$RUN_DIR/pruned/gpt.pth
if [[ ! -f "$PRUNED" ]]; then
    mkdir -p "$RUN_DIR/pruned"
    echo "=== pruning $(basename "$CKPT_SRC") -> pruned/gpt.pth ==="
    python tools/prune_gpt_checkpoint.py --input "$CKPT_SRC" --output "$PRUNED"
    echo "$(basename "$CKPT_SRC")" >"$RUN_DIR/pruned/SOURCE"
fi

OUT_DIR=$RUN_DIR/eval
mkdir -p "$OUT_DIR"

# With no PROMPT_WAV, prefer the curated clips in $REFS: they are already
# length- and alignment-checked, and unlike the corpus they do not depend on the
# dataset mount staying readable (tai8's permissions have gone away once already
# mid-project).
if [[ -z "$PROMPT_WAV" ]]; then
    for candidate in "$REFS"/ref_drama1_020_*.wav "$REFS"/*.wav; do
        if [[ -r "$candidate" ]]; then
            PROMPT_WAV=$candidate
            echo "=== using curated reference: $(basename "$PROMPT_WAV") ==="
            break
        fi
    done
fi

# Otherwise fall back to a well-aligned training clip. Selecting purely by
# duration would land on the corpus's worst tail: the clips over 10s average 0.67
# characters per second, i.e. mostly audio the transcript does not cover, which
# skews the generated speaking rate.
if [[ -z "$PROMPT_WAV" ]]; then
    PROMPT_WAV=$(python - <<'PY'
import json, os, re
han = re.compile(r"[一-鿿]")
work = os.environ["WORK"]
corpus = os.environ.get("CORPUS", "tai8")
for manifest in (
    f"{work}/{corpus}/shards/shard0/train_manifest.jsonl",
    f"{work}/shards/shard0/train_manifest.jsonl",
):
    if os.path.isfile(manifest):
        break
else:
    raise SystemExit("no shard manifest to pick a prompt from")
best = None
with open(manifest, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i > 40000:
            break
        r = json.loads(line)
        d = r.get("duration") or 0
        c = len(han.findall(r.get("text", "")))
        if not d or not c:
            continue
        rate = c / d
        if 3.0 <= d <= 8.0 and 4.0 <= rate <= 6.5:
            if best is None or d > best[0]:
                best = (d, r["audio"])
if best is None:
    raise SystemExit("no well-aligned clip found")
print(best[1])
PY
)
    echo "=== auto-selected prompt: $PROMPT_WAV ==="
fi

python - <<PY
import os
from indextts.infer_v2_modded import IndexTTS2

tts = IndexTTS2(
    cfg_path="$CKPT/config.yaml",
    model_dir="$CKPT",
    gpt_checkpoint_path="$PRUNED",
    bpe_model_path="$CKPT/bpe.model",
    use_fp16=True,
    use_deepspeed=False,
)
out = os.path.join("$OUT_DIR", "single.wav")
tts.infer(
    spk_audio_prompt="$PROMPT_WAV",
    text="""$TEXT""",
    output_path=out,
    verbose=True,
)
print("wrote", out)
PY
