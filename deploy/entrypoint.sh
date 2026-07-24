#!/usr/bin/env bash
set -euo pipefail

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/app/checkpoints}"
HF_REPO="${HF_REPO:-IndexTeam/IndexTTS-2}"

# Core weight files that must exist for the model to load. If any are missing,
# we (re)download the full repo into CHECKPOINTS_DIR before starting the webui.
REQUIRED_FILES=(
    "gpt.pth"
    "s2mel.pth"
    "wav2vec2bert_stats.pt"
    "bpe.model"
)

missing=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${CHECKPOINTS_DIR}/${f}" ]]; then
        echo "[entrypoint] Missing checkpoint file: ${f}"
        missing=1
    fi
done

if [[ "${missing}" -eq 1 ]]; then
    echo "[entrypoint] Downloading model weights from ${HF_REPO} into ${CHECKPOINTS_DIR} ..."
    # Prefer the new `hf` CLI; fall back to the legacy `huggingface-cli` on older
    # huggingface_hub versions. Both ship with the huggingface_hub package.
    if command -v hf >/dev/null 2>&1; then
        hf download "${HF_REPO}" --local-dir="${CHECKPOINTS_DIR}"
    else
        huggingface-cli download "${HF_REPO}" --local-dir "${CHECKPOINTS_DIR}"
    fi
    echo "[entrypoint] Download complete."
else
    echo "[entrypoint] All required checkpoint files present; skipping download."
fi

exec "$@"
