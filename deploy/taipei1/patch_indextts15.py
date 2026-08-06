#!/usr/bin/env python3
"""Make the IndexTTS 1.5 finetuning pipeline sample its conditioning per speaker.

The colleague's 1.5 run produced good Taiwanese but broken cloning: the output
did not resemble the reference, drifted to another voice partway through, and
could switch between a man and a woman. Two defects in that repo account for it,
and only the first has been fixed upstream.

  1. train.py used to blank the conditioning source whenever speaker_ids were
     present, so get_conditioning returned a learnable per-speaker embedding and
     the reference audio had no gradient path at all. At inference a new voice
     has no speaker id, so it fell back to the encoder — a distribution the LoRA
     had never been trained against. Fixed upstream in 3d667ca (2025-12-15);
     every call site now passes speaker_ids=None.

  2. _sample_condition_lazy still draws the conditioning clip from anywhere in
     the same manifest, on the assumption stated in its own docstring that one
     manifest is one speaker. The extraction tool writes manifests named
     audio_list_part_N holding an arbitrary slice of the corpus, so in practice
     the prompt is a different person nearly every time. A prompt that does not
     predict the target is a prompt the model learns to ignore — the same
     mechanism that cost v1 its emotion conditioning here.

This patches the second one, which needs a speaker to be carried end to end:

  * the audio_list parser accepts an optional third column
  * extract_codec records speaker_id alongside codes and mels
  * the lazy sampler buckets offsets by speaker instead of by manifest

Idempotent; re-running is a no-op.

  ROOT15=/mnt/shared/p06/indextts15 python deploy/taipei1/patch_indextts15.py
"""

from __future__ import annotations

import os
import sys

ROOT15 = os.environ.get("ROOT15", "/mnt/shared/p06/indextts15")
REPO = os.path.join(ROOT15, "repo")

EXTRACT = os.path.join(REPO, "tools", "extract_codec.py")
DATA_UTILS = os.path.join(REPO, "indextts", "data_utils.py")


def edit(path: str, old: str, new: str, label: str) -> bool:
    with open(path, encoding="utf-8") as handle:
        src = handle.read()
    if new in src:
        print(f"  [skip] {label} — already applied")
        return False
    if old not in src:
        raise SystemExit(f"  [FAIL] {label} — anchor not found in {path}")
    if src.count(old) != 1:
        raise SystemExit(f"  [FAIL] {label} — anchor appears {src.count(old)}x")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src.replace(old, new))
    print(f"  [ok]   {label}")
    return True


# 1. audio_list gains an optional speaker column.
edit(
    EXTRACT,
    """                    parts = line.split('\\t')
                    if len(parts) == 2:
                        wav_path, text = parts
                        if os.path.exists(wav_path):
                            self.samples.append({
                                'wav_path': wav_path,
                                'text': text
                            })""",
    """                    parts = line.split('\\t')
                    # A third column names the speaker. Without it every clip in
                    # a part looks like one speaker, and the conditioning ends up
                    # sampled from whoever happens to share the part.
                    if len(parts) in (2, 3):
                        wav_path, text = parts[0], parts[1]
                        speaker = parts[2] if len(parts) == 3 else None
                        if os.path.exists(wav_path):
                            self.samples.append({
                                'wav_path': wav_path,
                                'text': text,
                                'speaker': speaker
                            })""",
    "audio_list: accept a speaker column",
)

# 2. Carry it into the metadata the trainer reads.
edit(
    EXTRACT,
    """                            "audio": item['wav_path'],
                            "text": item['text'],
                            "codes": os.path.abspath(out_codebook),
                            "mels": os.path.abspath(out_mel),
                            "duration": round(duration, 4)""",
    """                            "audio": item['wav_path'],
                            "text": item['text'],
                            "codes": os.path.abspath(out_codebook),
                            "mels": os.path.abspath(out_mel),
                            "duration": round(duration, 4),
                            "speaker_id": item.get('speaker')""",
    "extract_codec: record speaker_id",
)

# 3. Sample the conditioning from the target's own speaker.
edit(
    DATA_UTILS,
    '''    def _sample_condition_lazy(self, manifest_idx: int, target_offset: int):
        """
        在 Lazy 模式下，隨機抽取同 Manifest (同 Speaker) 的另一條樣本當作 Conditioning。
        """
        offsets = self.manifest_offsets.get(manifest_idx, [])''',
    '''    def _lazy_speaker_offsets(self, manifest_idx: int, speaker_id):
        """Offsets in one manifest grouped by speaker.

        The offset index is built without parsing the lines, which is what makes
        it fast, so the speaker is not known there. Parse it once here, on first
        use, and keep the result: a few hundred thousand short JSON lines take
        seconds and the map is small.
        """
        if speaker_id is None:
            return None
        cache = getattr(self, "_lazy_spk_cache", None)
        if cache is None:
            cache = self._lazy_spk_cache = {}
        buckets = cache.get(manifest_idx)
        if buckets is None:
            import json as _json

            buckets = {}
            manifest_file = self.manifest_files[manifest_idx]
            for off in self.manifest_offsets.get(manifest_idx, []):
                try:
                    item = self._read_item(manifest_file, off)
                except Exception:
                    continue
                sid = item.get("speaker_id")
                if sid:
                    buckets.setdefault(sid, []).append(off)
            cache[manifest_idx] = buckets
        return buckets.get(speaker_id)

    def _sample_condition_lazy(self, manifest_idx: int, target_offset: int, speaker_id=None):
        """Draw the conditioning clip from another utterance by the same speaker.

        This used to draw from the whole manifest, on the assumption that one
        manifest is one speaker. Extraction writes manifests named
        audio_list_part_N holding an arbitrary slice of the corpus, so that
        assumption is false and the prompt was almost always somebody else —
        which teaches the model that the prompt does not predict the target, and
        so to ignore it.
        """
        offsets = self._lazy_speaker_offsets(manifest_idx, speaker_id)
        if not offsets:
            offsets = self.manifest_offsets.get(manifest_idx, [])''',
    "data_utils: bucket lazy conditioning by speaker",
)

edit(
    DATA_UTILS,
    "            cond_spec = self._sample_condition_lazy(manifest_idx, offset)",
    "            cond_spec = self._sample_condition_lazy(manifest_idx, offset, speaker_id)",
    "data_utils: pass the speaker to the sampler",
)

# The in-memory path already buckets by speaker, but falls back to inferring one
# from the audio path when the field is absent; with the field present it is used
# directly, so nothing there needs changing.

print("\ndone. verify with:")
print(f"  cd {REPO} && PYTHONPATH={REPO}:{ROOT15}/pylibs python -c 'import indextts.data_utils'")
