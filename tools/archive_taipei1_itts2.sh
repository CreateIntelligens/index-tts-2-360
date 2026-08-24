#!/usr/bin/env bash
# Archive the IndexTTS 2 line off Taipei-1, on the same terms as the 1.5 one:
# something that can still be run for inference, the settings that produced it,
# and the records of what was actually tested.
#
# Each of the five runs holds four 7.7 GB training checkpoints (latest.pth plus
# three model_step*.pth) and one 3.48 GB pruned/gpt.pth. Only the pruned one is
# taken. The other four are full training state — optimiser included — and are
# useful for exactly one thing, resuming that run on the machine being handed
# back. Skipping them turns 171 GB into 17.4 GB.
#
# Also skipped: checkpoints/ (the official base weights, still on HuggingFace)
# and images/indextts2-vllm.sqsh (17 GB, rebuilt from the Dockerfile).
#
# Kept small but deliberately: eval/ holds the listening samples that every
# judgement about v1..v4 was made from, and refs/ holds the reference wavs those
# samples were cloned from. Neither can be regenerated once the weights that
# produced them are gone.
#
#   bash tools/archive_taipei1_itts2.sh
#   DRY=1 bash tools/archive_taipei1_itts2.sh

set -uo pipefail

# Host, account and key come from the environment; see the file for what to set.
. "$(dirname "${BASH_SOURCE[0]}")/taipei1_env.sh"

SP=${SP:-$(mktemp -d)}
REMOTE=/mnt/shared/p06/indextts2
DEST=${DEST:-/mnt/nas/indextts2_taipei_1_no5}
REL=${DEST#/mnt/nas/}
DRY=${DRY:-0}
PAR=${PAR:-4}
IMAGE=nas-xfer
RUNS="v1_20260729_191628 v2_20260730_104546 v3a_20260730_154608 v3b_20260730_172820 v4_20260805_114114"

in_container() {
    docker run --rm -v "$KEYDIR":/keys:ro -v /mnt/nas:/nas "$IMAGE" sh -c "$1"
}

cat > "$SP/Dockerfile.xfer" <<'EOF'
FROM alpine:3.20
RUN apk add --no-cache openssh-client rsync
EOF
docker build -q -t "$IMAGE" -f "$SP/Dockerfile.xfer" "$SP" >/dev/null
in_container "mkdir -p /nas/$REL"

echo "=== 推論權重：5 顆 pruned/gpt.pth，各 3.48 GB ==="
fetch_one() {
    run=$1
    dst=/mnt/nas/$REL/$run/gpt.pth
    # 3484568465 or 3484623454 depending on the run; any file over 3 GB is the
    # real thing rather than a truncated retry.
    if [ -f "$dst" ] && [ "$(stat -c %s "$dst" 2>/dev/null)" -gt 3000000000 ]; then
        echo "  [skip] $run"; return 0
    fi
    [ "$DRY" = "1" ] && { echo "  [would] $run"; return 0; }
    for attempt in 1 2 3 4; do
        if docker run --rm -v "$KEYDIR":/keys:ro -v /mnt/nas:/nas nas-xfer sh -c \
            "mkdir -p /nas/$REL/$run && scp $SSH_OPTS_CONT -i /keys/$KEYNAME -P $TP1_PORT \
             -o 'ProxyCommand=ssh $SSH_OPTS_CONT -i /keys/$KEYNAME -W %h:%p -p $TP1_PORT $TP1_JUMP' \
             '$RHOST:$REMOTE/$run/pruned/gpt.pth' '/nas/$REL/$run/gpt.pth'" >/dev/null 2>&1; then
            echo "  [ok]   $run"; return 0
        fi
        sleep $((attempt * 5))
    done
    echo "  [FAIL] $run"
}
RHOST=$TP1_HOST
export -f fetch_one
export REL REMOTE RHOST DRY KEYDIR KEYNAME TP1_PORT TP1_JUMP SSH_OPTS_CONT
echo "$RUNS" | tr ' ' '\n' | xargs -P "$PAR" -n 1 bash -c 'fetch_one "$0"'

echo "=== 設定、聽測樣本、log、參考音檔 ==="
# One tar rather than a file at a time: these are hundreds of small files and
# the jump host refuses connections when several arrive at once.
if [ "$DRY" != "1" ]; then
    tp1 "cd $REMOTE && tar -cf - \
        v*/run_config.json v*/eval v*/logs v*/pruned/SOURCE v*/train.log \
        analysis tools refs logs 2>/dev/null" > "$SP/itts2_meta.tar" 2>/dev/null
    ls -la "$SP/itts2_meta.tar" | awk '{printf "  打包 %.1f MB\n", $5/1e6}'
    docker run --rm -i -v /mnt/nas:/nas "$IMAGE" \
        sh -c "mkdir -p /nas/$REL/meta && tar -xf - -C /nas/$REL/meta" < "$SP/itts2_meta.tar"
    rm -f "$SP/itts2_meta.tar"
fi

echo
echo "=== 結果 ==="
in_container "du -sh /nas/$REL 2>/dev/null
echo -n '  推論權重: '; find /nas/$REL -name 'gpt.pth' | wc -l
echo -n '  meta 檔數: '; find /nas/$REL/meta -type f 2>/dev/null | wc -l"
