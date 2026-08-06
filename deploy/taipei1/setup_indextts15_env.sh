#!/usr/bin/env bash
# Stand up an IndexTTS 1.5 environment on Taipei-1 without building a new image.
#
# The 1.5 project pins transformers 4.44 and peft 0.13, while the container
# carries transformers 4.57. Those pins turned out not to matter: 1.5 imports
# GPT2Config/GPT2Model and get_cosine_schedule_with_warmup, all of which are
# still public in 4.57, and it does not vendor a copy of transformers the way
# this repo does — so nothing depends on the symbols that were removed. A
# current peft works against 4.57; the pinned 0.13 would not.
#
# What is genuinely missing from the image is seven small pure-Python packages.
# They go in a pylibs directory of their own rather than the one IndexTTS2 uses,
# because that one is on PYTHONPATH for every training and preprocessing job and
# a stray top-level package there would shadow a real module. pip installing
# cn2an with --target, for instance, drops `build/`, `src/` and `example/` next
# to it — three of the most collision-prone names there are.
#
# Idempotent; safe to re-run.
#
#   bash deploy/taipei1/setup_indextts15_env.sh

set -euo pipefail

ROOT15=${ROOT15:-/mnt/shared/p06/indextts15}
PYLIBS15=$ROOT15/pylibs
# Deliberately not the IndexTTS2 pylibs. Keep the two apart.
SHARED_PYLIBS=/mnt/shared/p06/indextts2/pylibs

PACKAGES=(peft loguru cn2an g2p-en vocos ffmpeg-python proces)

mkdir -p "$PYLIBS15"

echo "=== installing into $PYLIBS15 ==="
# --no-deps: everything else these pull in (torch, transformers, numpy,
# safetensors, accelerate) is already in the image at versions the rest of the
# pipeline is tested against, and --target would happily shadow them.
pip install --quiet --target="$PYLIBS15" --upgrade --no-deps "${PACKAGES[@]}"

# cn2an ships its repo layout, not just its package.
rm -rf "$PYLIBS15/build" "$PYLIBS15/src" "$PYLIBS15/example"

echo "=== $PYLIBS15 ==="
ls "$PYLIBS15" | grep -v dist-info || true

echo
echo "=== IndexTTS2's pylibs must still contain only opencc ==="
ls "$SHARED_PYLIBS"

cat <<EOF

Run 1.5 with:

  cd $ROOT15/repo
  PYTHONPATH=$ROOT15/repo:$PYLIBS15 python ...

The cd matters: the image bundles its own indextts at /app, and it wins unless
the working directory puts the checkout first.
EOF
