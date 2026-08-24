#!/usr/bin/env bash
# The one thing still missing from the IndexTTS archives after the 08-24 audit.
# CosyVoice is a separate line handled elsewhere and is not touched here.
#
# indextts2-vllm.sqsh
#   17.4 GB. The matching local image (indextts2-serve:latest) was never pushed
#   to a registry, so this machine is the only copy. The CosyVoice image is
#   deliberately not archived: that one does live in a registry, so a NAS copy
#   would be the third.
#
#   bash tools/archive_taipei1_gaps.sh

set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/taipei1_env.sh"

SP=${SP:-$(mktemp -d)}
IMAGE=nas-xfer
RHOST=$TP1_HOST

fetch() {   # remote-path  nas-dir  [expected-size]
    path=$1; dest=$2; want=${3:-}
    name=$(basename "$path")
    if [ -f "/mnt/nas/$dest/$name" ]; then
        have=$(stat -c %s "/mnt/nas/$dest/$name" 2>/dev/null)
        [ -n "$want" ] && [ "$have" = "$want" ] && { echo "  [skip] $name"; return 0; }
    fi
    for attempt in 1 2 3 4; do
        if docker run --rm -v "$KEYDIR":/keys:ro -v /mnt/nas:/nas "$IMAGE" sh -c \
            "mkdir -p /nas/$dest && scp $SSH_OPTS_CONT -i /keys/$KEYNAME -P $TP1_PORT \
             -o 'ProxyCommand=ssh $SSH_OPTS_CONT -i /keys/$KEYNAME -W %h:%p -p $TP1_PORT $TP1_JUMP' \
             '$RHOST:$path' '/nas/$dest/$name'" >/dev/null 2>&1; then
            echo "  [ok]   $name"; return 0
        fi
        sleep $((attempt * 10))
    done
    echo "  [FAIL] $name"
}

echo "=== indextts2-vllm.sqsh ==="
fetch /mnt/shared/p06/indextts2/images/indextts2-vllm.sqsh \
      indextts2_taipei_1_no5/images 17378009088

echo
echo "=== 結果 ==="
docker run --rm -v /mnt/nas:/nas "$IMAGE" sh -c '
for d in /nas/indextts2_taipei_1_no5/images; do
  echo "  $d"; ls -la $d 2>/dev/null | tail -n +4 | awk "{printf \"    %12s %s\n\", \$5, \$9}"
done'
