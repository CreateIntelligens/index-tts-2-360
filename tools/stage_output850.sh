#!/usr/bin/env bash
# Ship the 850 episodes of segment_data/output that tai8 does not cover up to
# Taipei-1.
#
# Three things make a naive `rsync /mnt/nas/.../output` the wrong move:
#
#   * 44.1 kHz is wasted bandwidth. Everything downstream resamples to 24 kHz
#     (MelSpectrogramFeatures) or 16 kHz (speaker embeddings), so 24 kHz mono
#     FLAC is lossless as far as the pipeline is concerned and measures 0.35x
#     the size — 119 GB becomes about 42 GB, and at the 4.2 MB/s this link gives
#     that is the difference between 8 hours and 3.
#   * 1.17 million files (wav + transcript) would spend most of the transfer in
#     rsync's per-file round trips. They go up as one tar per episode instead:
#     850 files, each still individually resumable.
#   * The 242 episodes tai8 already covers are left out on purpose. Their audio
#     is in both corpora under different cuts, and output's transcripts are the
#     uncorrected ones — the inverted Taiwanese word order that the workers were
#     paid to normalise. Feeding both would teach the model two contradictory
#     readings of the same line.
#
# Source is read-only throughout; everything lands in $STAGE, a new directory.
# Restartable: an episode whose tar already exists is skipped.
#
#   bash tools/stage_output850.sh          # convert, tar, then upload
#   PHASE=convert bash tools/stage_output850.sh
#   PHASE=upload  bash tools/stage_output850.sh

set -euo pipefail

# Host, account and key come from the environment; see the file for what to set.
. "$(dirname "${BASH_SOURCE[0]}")/taipei1_env.sh"

SRC=${SRC:-/mnt/nas/ml-material/segment_data/output}
TAI8=${TAI8:-/mnt/nas/dataset202607_1/tai8/manifests/index_tts}
STAGE=${STAGE:-$HOME/taigi_output850_24k}
# A top level directory of its own, kept clear of the indextts2 tree.
REMOTE=${REMOTE:-/mnt/shared/p06/taigi_output850}
JOBS=${JOBS:-14}
PHASE=${PHASE:-all}

SSH_CMD=$SSH_CMD_HOST
RHOST=$TP1_HOST

mkdir -p "$STAGE/tars" "$STAGE/lists"

# ---------------------------------------------------------------- episode list
if [ ! -s "$STAGE/lists/episodes.txt" ]; then
    echo "=== working out which episodes tai8 does not cover ==="
    python3 - "$SRC" "$TAI8" "$STAGE" <<'PY'
import collections, json, os, re, sys

src, tai8_dir, stage = sys.argv[1:4]
# tai8 paths are segments/<drama>/<speaker>/<episode>/..., so the episode is the
# third component, not the second.
rx = re.compile(r"segments/(drama\d)/[^/]+/([^/]+)/")
covered = collections.defaultdict(set)
for split in ("train", "val"):
    path = f"{tai8_dir}/{split}_manifest.jsonl"
    if not os.path.isfile(path):
        continue
    for line in open(path, encoding="utf-8"):
        m = rx.search(json.loads(line)["audio"])
        if m:
            covered[m.group(1)].add(m.group(2))

out = []
for drama in ("drama1", "drama2"):
    episodes = set()
    root = os.path.join(src, drama)
    for speaker in os.listdir(root):
        speaker_dir = os.path.join(root, speaker)
        if os.path.isdir(speaker_dir):
            episodes.update(os.listdir(speaker_dir))
    keep = sorted(episodes - covered[drama])
    print(f"  {drama}: {len(episodes)} 集，扣掉 tai8 的 {len(covered[drama])} 集，留下 {len(keep)}")
    out += [f"{drama}\t{e}" for e in keep]

with open(f"{stage}/lists/episodes.txt", "w") as handle:
    handle.write("\n".join(out) + "\n")
print(f"  total {len(out)} episodes")
PY
fi
TOTAL=$(wc -l < "$STAGE/lists/episodes.txt")

# -------------------------------------------------------------------- convert
convert_episode() {
    local drama=$1 ep=$2
    local tar="$STAGE/tars/${drama}__${ep}.tar"
    [ -f "$tar" ] && return 0

    local work="$STAGE/work/${drama}__${ep}"
    rm -rf "$work"
    # Layout is <speaker>/<episode>/, so an episode's clips are scattered across
    # every speaker directory. Re-nested here as <episode>/<speaker>/ because
    # speaker ids are scoped per episode anyway — pyannote's cross-episode
    # linking measured 18.13% pairwise precision and is not trusted.
    for speaker_dir in "$SRC/$drama"/*/"$ep"; do
        [ -d "$speaker_dir" ] || continue
        local speaker
        speaker=$(basename "$(dirname "$speaker_dir")")
        mkdir -p "$work/$speaker"
        for wav in "$speaker_dir"/*.wav; do
            [ -e "$wav" ] || continue
            local base txt
            base=$(basename "$wav" .wav)
            txt="$speaker_dir/$base.normalized.txt"
            # A clip with no transcript is unusable; skip the pair entirely
            # rather than ship audio that the manifest builder will drop.
            [ -s "$txt" ] || continue
            ffmpeg -v error -nostdin -y -i "$wav" \
                -ar 24000 -ac 1 -c:a flac -compression_level 5 \
                "$work/$speaker/$base.flac" </dev/null || continue
            cp "$txt" "$work/$speaker/$base.txt"
        done
        rmdir "$work/$speaker" 2>/dev/null || true
    done

    if [ -d "$work" ] && [ -n "$(ls -A "$work" 2>/dev/null)" ]; then
        tar -cf "$tar.part" -C "$STAGE/work" "${drama}__${ep}"
        mv "$tar.part" "$tar"
    fi
    rm -rf "$work"
}
export -f convert_episode
export SRC STAGE

if [ "$PHASE" = all ] || [ "$PHASE" = convert ]; then
    echo "=== converting to 24 kHz mono FLAC, $JOBS at a time ==="
    echo "    $TOTAL episodes -> $STAGE/tars"
    tr '\t' ' ' < "$STAGE/lists/episodes.txt" \
        | xargs -P "$JOBS" -n 2 bash -c 'convert_episode "$0" "$1"'
    echo "    done: $(ls "$STAGE/tars" | grep -c '\.tar$')/$TOTAL tars, $(du -sh "$STAGE/tars" | cut -f1)"
    # Tells the uploader running alongside that no more tars are coming.
    touch "$STAGE/.convert_done"
fi

# --------------------------------------------------------------------- upload
if [ "$PHASE" = all ] || [ "$PHASE" = upload ]; then
    echo "=== uploading to $REMOTE ==="
    eval "$SSH_CMD $RHOST 'mkdir -p $REMOTE/tars'"
    # --partial so a dropped connection resumes mid-tar instead of restarting
    # the file; no -z because FLAC is already compressed. Excluding *.part keeps
    # a tar that is still being written out of the transfer — the converter
    # writes <name>.tar.part and renames it, so only the rename makes it visible.
    #
    # PRUNE=1 hands the local copy over: rsync deletes each source file once it
    # has verified the transfer, which caps the staging directory at whatever
    # the converter is ahead by instead of accumulating all 35 GB. The cost is
    # that a tar found bad on the far side would have to be reconverted.
    rsync -a --info=progress2 --partial --human-readable \
        --exclude '*.part' ${PRUNE:+--remove-source-files} \
        -e "$SSH_CMD" \
        "$STAGE/tars/" "$RHOST:$REMOTE/tars/"
    eval "$SSH_CMD $RHOST 'ls $REMOTE/tars | wc -l; du -sh $REMOTE/tars'"
    echo "=== upload finished ==="
fi
