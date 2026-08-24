#!/usr/bin/env python3
"""Turn the corpora into audio_list parts for the IndexTTS 1.5 tools.

The 1.5 extraction tool wants `audio_list_part_N.txt`, one line per clip. Two
columns is what it shipped with; a third naming the speaker is added by
patch_indextts15.py, because without it a part looks like a single speaker and
the trainer draws each conditioning clip from whoever else lands in the same
part — the defect that cost the earlier 1.5 run its cloning.

Speakers are kept whole within a part. The conditioning sampler picks another
clip by the same speaker from the same manifest, so splitting a speaker across
parts would shrink the pool it can draw from for no benefit.

Two kinds of source:

  manifests   tai8 and naer, read from their index_tts jsonl.
  output850   the 850 episodes of segment_data/output that tai8 does not cover,
              laid out as <drama>__<episode>/<speaker>/<clip>.flac with the
              transcript beside it. Speaker ids here are pyannote's, and they
              are scoped to the episode rather than used globally: pyannote's
              cross-episode linking measured 18.13% pairwise precision, so a
              global id merges people who are not the same, and a wrong pair
              teaches the model to disregard the conditioning. Scoping throws
              those links away deliberately. A real actor becomes many
              pseudo-speakers, which costs nothing — pairing only needs
              "same person" to be true, never "same person across episodes".

Durations for output850 come from a side file rather than the audio, because
opening 586,215 files on Lustre to read a header is slow and the sizes were
already known from the source wavs.

  OUT_DIR=... PARTS=4 python tools/build_indextts15_audio_lists.py
"""

from __future__ import annotations

import collections
import json
import os

DATASET = os.environ.get("DATASET", "/mnt/shared/p06/dataset202607_1")
CORPORA = os.environ.get("CORPORA", "tai8 naer").split()
SPLITS = os.environ.get("SPLITS", "train").split()
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/shared/p06/indextts15/finetune_data/audio_list")
PARTS = int(os.environ.get("PARTS", "4"))
MIN_DUR = float(os.environ.get("MIN_DUR", "1.0"))
MAX_DUR = float(os.environ.get("MAX_DUR", "15.0"))
# Empty disables it; the v4 comparison runs are rebuilt without output850.
OUTPUT850 = os.environ.get("OUTPUT850", "")
OUTPUT850_DURATIONS = os.environ.get("OUTPUT850_DURATIONS", "")


def add_output850(by_speaker: dict[str, list[tuple[str, str]]]) -> tuple[int, int]:
    """Walk the extracted tree, pairing each clip with the text beside it."""
    durations = {}
    if OUTPUT850_DURATIONS and os.path.isfile(OUTPUT850_DURATIONS):
        for line in open(OUTPUT850_DURATIONS, encoding="utf-8"):
            rel, seconds = line.rstrip("\n").split("\t")
            durations[rel] = float(seconds)
        print(f"  output850: {len(durations):,} durations loaded")
    else:
        print("  [warn] no duration file — length filtering is off for output850")

    kept = dropped = 0
    for episode in sorted(os.listdir(OUTPUT850)):
        episode_dir = os.path.join(OUTPUT850, episode)
        if not os.path.isdir(episode_dir):
            continue
        for speaker in sorted(os.listdir(episode_dir)):
            speaker_dir = os.path.join(episode_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            # Episode-scoped, and namespaced so it can never collide with a
            # tai8 or naer label.
            label = f"out850_{episode}_{speaker}"
            for name in sorted(os.listdir(speaker_dir)):
                if not name.endswith(".flac"):
                    continue
                base = name[:-5]
                rel = f"{episode}/{speaker}/{base}"
                seconds = durations.get(rel)
                if seconds is not None and not MIN_DUR <= seconds <= MAX_DUR:
                    dropped += 1
                    continue
                text_path = os.path.join(speaker_dir, base + ".txt")
                try:
                    with open(text_path, encoding="utf-8") as handle:
                        text = handle.read().replace("\t", " ").strip()
                except OSError:
                    dropped += 1
                    continue
                if not text:
                    dropped += 1
                    continue
                by_speaker[label].append((os.path.join(speaker_dir, name), text))
                kept += 1
    return kept, dropped


def main() -> None:
    by_speaker: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    missing = skipped = 0

    for corpus in CORPORA:
        manifest_dir = f"{DATASET}/{corpus}/manifests/index_tts"
        for split in SPLITS:
            path = f"{manifest_dir}/{split}_manifest.jsonl"
            if not os.path.isfile(path):
                print(f"  [skip] no {path}")
                continue
            n = 0
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                if not MIN_DUR <= d.get("duration", 0) <= MAX_DUR:
                    skipped += 1
                    continue
                # Manifest paths are relative to the manifest directory.
                audio = os.path.normpath(os.path.join(manifest_dir, d["audio"]))
                if not os.path.isfile(audio):
                    missing += 1
                    continue
                text = d["text"].replace("\t", " ").strip()
                if not text:
                    skipped += 1
                    continue
                # Namespace the speaker so two corpora cannot collide on a label.
                by_speaker[f"{corpus}_{d['speaker']}"].append((audio, text))
                n += 1
            print(f"  {corpus}/{split}: {n:,} clips")

    if OUTPUT850:
        kept, dropped = add_output850(by_speaker)
        print(f"  output850: {kept:,} clips ({dropped:,} dropped)")

    if missing:
        print(f"  [warn] {missing:,} clips missing on disk")
    if skipped:
        print(f"  [warn] {skipped:,} clips outside {MIN_DUR}-{MAX_DUR}s or with empty text")

    # Greedy longest-first packing keeps the parts close in size, which matters
    # because one extraction process is launched per part.
    order = sorted(by_speaker, key=lambda s: -len(by_speaker[s]))
    buckets: list[list[str]] = [[] for _ in range(PARTS)]
    counts = [0] * PARTS
    for spk in order:
        i = counts.index(min(counts))
        buckets[i].append(spk)
        counts[i] += len(by_speaker[spk])

    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for i, spks in enumerate(buckets):
        path = f"{OUT_DIR}/audio_list_part_{i}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            for spk in spks:
                for audio, text in by_speaker[spk]:
                    handle.write(f"{audio}\t{text}\t{spk}\n")
                    total += 1
        print(f"  part {i}: {counts[i]:,} clips, {len(spks):,} speakers -> {path}")

    print(f"\n  {total:,} clips from {len(by_speaker):,} speakers into {PARTS} parts")


if __name__ == "__main__":
    main()
