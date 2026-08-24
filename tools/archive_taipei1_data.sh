#!/usr/bin/env bash
# Archive the processed corpora from Taipei-1 alongside the models.
#
# Both of these exist in raw form on the NAS already; what only exists on
# Taipei-1 is the processed version, and reproducing it costs hours:
#
#   taigi_output850   850 episode tars, 24 kHz mono FLAC. The source wavs are
#                     on the NAS at 44.1 kHz, so regenerating means about 100
#                     minutes of ffmpeg over 586,215 files.
#   suisiann_tailo    the dual-script tree, three writings of every clip. Its
#                     three directories are hard links to one another, so it is
#                     tarred on the far side rather than copied — rsync without
#                     -H would land three real copies of 800 MB.
#
# Same container trick as archive_taipei1.sh: the NAS is uid=0 and this account
# cannot write to it, docker can, and the data never touches local disk.
#
#   bash tools/archive_taipei1_data.sh

set -uo pipefail

# Host, account and key come from the environment; see the file for what to set.
. "$(dirname "${BASH_SOURCE[0]}")/taipei1_env.sh"

SP=${SP:-$(mktemp -d)}
RHOST=$TP1_HOST
DEST=${DEST:-/mnt/nas/indextts15_taipei_1_no5}
REL=${DEST#/mnt/nas/}
PAR=${PAR:-2}
IMAGE=nas-xfer

in_container() {
    docker run --rm -v "$KEYDIR":/keys:ro -v /mnt/nas:/nas "$IMAGE" sh -c "$1"
}

fetch_one() {   # size remote-path dest-subdir
    size=$1; path=$2; sub=$3
    name=$(basename "$path")
    # Two spellings of the same file: the host sees /mnt/nas, the container
    # sees /nas. Passing the host path into the container is what made every
    # transfer fail silently the first time.
    host_dst=/mnt/nas/$REL/$sub/$name
    cont_dst=/nas/$REL/$sub/$name
    if [ -f "$host_dst" ] && [ "$(stat -c %s "$host_dst" 2>/dev/null)" = "$size" ]; then
        return 0
    fi
    # Retried, because the jump host refuses connections when several arrive at
    # once. A single transfer of a file that had just "failed" succeeded on its
    # own, so the failures are contention, not bad files. 850 small files churn
    # connections far faster than 52 large ones did.
    for attempt in 1 2 3 4; do
        if docker run --rm -v "$KEYDIR":/keys:ro -v /mnt/nas:/nas nas-xfer sh -c \
            "mkdir -p /nas/$REL/$sub && scp $SSH_OPTS_CONT -i /keys/$KEYNAME -P $TP1_PORT \
         -o 'ProxyCommand=ssh $SSH_OPTS_CONT -i /keys/$KEYNAME -W %h:%p -p $TP1_PORT $TP1_JUMP' \
             '$RHOST:$path' '$cont_dst'" >/dev/null 2>&1; then
            return 0
        fi
        sleep $((attempt * 5))
    done
    echo "  [FAIL] $sub/$name"
}
export -f fetch_one
export REL RHOST KEYDIR KEYNAME TP1_PORT TP1_JUMP SSH_OPTS_CONT

echo "=== taigi_output850：850 個 tar ==="
tp1 "find /mnt/shared/p06/taigi_output850 -maxdepth 2 -type f -exec stat -c '%s %n' {} +" \
    > "$SP/data_manifest.txt" 2>/dev/null
printf "  %d 個檔案，%.1f GB\n" "$(wc -l < "$SP/data_manifest.txt")" \
    "$(awk '{s+=$1} END {print s/1e9}' "$SP/data_manifest.txt")"

awk '{print $1, $2, "output850"}' "$SP/data_manifest.txt" \
    | xargs -P "$PAR" -n 3 bash -c 'fetch_one "$0" "$1" "$2"'

echo "=== suisiann_tailo：遠端打包以保住硬連結 ==="
# tar records the second and third link as links, so the archive stays near the
# size of one copy instead of three.
tp1 "cd /mnt/shared/p06 && tar -cf - suisiann_tailo | gzip -1" \
    > "$SP/suisiann_tailo.tgz" 2>/dev/null
ls -la "$SP/suisiann_tailo.tgz" | awk '{printf "  打包 %.0f MB\n", $5/1e6}'
in_container "mkdir -p /nas/$REL/suisiann_tailo"
in_container "cat > /nas/$REL/suisiann_tailo/suisiann_tailo.tgz" < "$SP/suisiann_tailo.tgz"
rm -f "$SP/suisiann_tailo.tgz"

echo
echo "=== 結果 ==="
in_container "du -sh /nas/$REL/output850 /nas/$REL/suisiann_tailo 2>/dev/null
echo -n '  output850 檔案數: '; ls /nas/$REL/output850 | wc -l"
