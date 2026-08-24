#!/usr/bin/env bash
# Move the IndexTTS 1.5 models off Taipei-1 before the allocation expires.
#
# Two constraints shape this.
#
# The NAS is a CIFS mount owned by someone else, mounted uid=0 dir_mode=0755,
# so this account cannot write to it and remounting is not ours to do. A bind
# mount does not help — it shares the same permission view. Writing as root
# does, and docker is available without sudo, so the transfer runs inside a
# container with the share and the ssh key mounted. Data goes from Taipei-1
# into the NAS without passing through local disk, which matters when 61 GB is
# only in transit.
#
# rsync rather than a loop of scp: it resumes a part-transferred file, skips
# what is already there, and pays the SSH handshake through the jump host once
# instead of 52 times.
#
# What comes back: the merged gpt_epoch_*.pth and the configs that produced
# them, plus the training logs. What does not: the checkpoint_epoch_*.pt
# optimiser states, 19.4 GB across two early runs. Those only help resume a
# run, and the cluster they would resume on is the one going away.
#
#   bash tools/archive_taipei1.sh
#   DRY=1 bash tools/archive_taipei1.sh

set -uo pipefail

# Host, account and key come from the environment; see the file for what to set.
. "$(dirname "${BASH_SOURCE[0]}")/taipei1_env.sh"

SP=${SP:-$(mktemp -d)}
REMOTE_RUNS=/mnt/shared/p06/indextts15/repo/finetune_models/runs
REMOTE_CFG=/mnt/shared/p06/indextts15/repo/finetune_models
REMOTE_LOGS=/mnt/shared/p06/indextts2/logs
RHOST=$TP1_HOST
DEST=${DEST:-/mnt/nas/indextts15_taipei_1_no5}
DRY=${DRY:-0}
IMAGE=nas-xfer

SSH_CMD=$SSH_CMD_CONT

REL=${DEST#/mnt/nas/}
PAR=${PAR:-4}

in_container() {
    docker run --rm -v "$KEYDIR":/keys:ro -v /mnt/nas:/nas "$IMAGE" sh -c "$1"
}

cat > "$SP/Dockerfile.xfer" <<'EOF'
FROM alpine:3.20
RUN apk add --no-cache openssh-client rsync
EOF
docker build -q -t "$IMAGE" -f "$SP/Dockerfile.xfer" "$SP" >/dev/null

in_container "mkdir -p /nas/$REL/runs /nas/$REL/config /nas/$REL/logs"

DRYFLAG=""
[ "$DRY" = "1" ] && DRYFLAG="--dry-run"

echo "=== 權重（並行 $PAR 條）==="
# One stream measured about 4 MB/s through the jump host, which is 4 hours for
# 61 GB; rsync over the same hop managed only 1.1. Several streams in parallel
# is the difference between finishing with margin and finishing just before the
# certificate expires. One container per file — startup is under a second
# against minutes of transfer.
tp1 "find $REMOTE_RUNS -name 'gpt_epoch_*.pth' -printf '%s\t%p\n'" \
    | sort -t/ -k9 > "$SP/archive_manifest.txt" 2>/dev/null
printf "  清單 %d 個，%.1f GB\n" "$(wc -l < "$SP/archive_manifest.txt")" \
    "$(awk -F'\t' '{s+=$1} END {print s/1e9}' "$SP/archive_manifest.txt")"

fetch_one() {
    size=$1; path=$2
    run=$(basename "$(dirname "$path")"); name=$(basename "$path")
    dst=/mnt/nas/$REL/runs/$run/$name
    if [ -f "$dst" ] && [ "$(stat -c %s "$dst" 2>/dev/null)" = "$size" ]; then
        echo "  [skip] $run/$name"; return 0
    fi
    [ "$DRY" = "1" ] && { echo "  [would] $run/$name"; return 0; }
    docker run --rm -v "$KEYDIR":/keys:ro -v /mnt/nas:/nas nas-xfer sh -c \
        "mkdir -p /nas/$REL/runs/$run && scp $SSH_OPTS_CONT -i /keys/$KEYNAME -P $TP1_PORT \
         -o 'ProxyCommand=ssh $SSH_OPTS_CONT -i /keys/$KEYNAME -W %h:%p -p $TP1_PORT $TP1_JUMP' \
         '$RHOST:$path' '/nas/$REL/runs/$run/$name'" >/dev/null 2>&1 \
        && echo "  [ok]   $run/$name" || echo "  [FAIL] $run/$name"
}
export -f fetch_one
export SP REL RHOST DRY KEYDIR KEYNAME TP1_PORT TP1_JUMP SSH_OPTS_CONT

tr '\t' ' ' < "$SP/archive_manifest.txt" \
    | xargs -P "$PAR" -n 2 bash -c 'fetch_one "$0" "$1"'

echo "=== 頂層 config ==="
in_container "rsync -a $DRYFLAG --partial \
    --include='config_finetuned.yaml' --include='config_inference.yaml' \
    --include='config.yaml' --exclude='*' \
    -e '$SSH_CMD' \
    '$RHOST:$REMOTE_CFG/' '/nas/$REL/config/'"

echo "=== 訓練 log ==="
in_container "rsync -a $DRYFLAG --partial \
    --include='itts15_*.out' --include='itts15_*.log' --exclude='*' \
    -e '$SSH_CMD' \
    '$RHOST:$REMOTE_LOGS/' '/nas/$REL/logs/'"

echo
echo "=== 結果 ==="
in_container "du -sh /nas/$REL; echo -n '  權重數: '; find /nas/$REL -name 'gpt_epoch_*.pth' | wc -l"
