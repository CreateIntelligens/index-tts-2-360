#!/usr/bin/env python3
"""
Drop manifest entries whose text cannot be represented by the tokenizer.

The IndexTTS2 base vocab has no byte fallback, so any character missing from it
encodes to a single <unk>. Leaving those entries in teaches the model that <unk>
maps to arbitrary audio, so it is cleaner to remove them up front.

Reads a source manifest and writes a filtered copy; the source is never modified.
Run this *before* `tools/preprocess_data.py` and point preprocessing at the output.

Example:
    python tools/filter_manifest.py \
        --manifest /path/to/tai8/manifests/index_tts/train_manifest.jsonl \
        --output   /path/to/work/train_manifest.filtered.jsonl \
        --tokenizer checkpoints/bpe.model \
        --language zh --zh-to-simplified
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from indextts.utils.front import TextNormalizer, TextTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter manifest entries that tokenize to <unk>.")
    parser.add_argument("--manifest", type=Path, required=True, help="Source JSONL manifest (read-only).")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the filtered manifest.")
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/bpe.model"), help="SentencePiece model.")
    parser.add_argument("--language", type=str, default=None, help="Language hint for the normalizer (e.g. zh).")
    parser.add_argument(
        "--zh-to-simplified",
        action="store_true",
        help="Fold Traditional Chinese onto Simplified. Must match the preprocessing/inference setting.",
    )
    parser.add_argument(
        "--text-field", type=str, default="text", help="Record field holding the transcript."
    )
    parser.add_argument(
        "--audio-field", type=str, default="audio", help="Record field holding the audio path."
    )
    parser.add_argument(
        "--keep-relative-audio",
        action="store_true",
        help=(
            "Leave the audio path untouched. By default it is rewritten to an absolute "
            "path resolved against the SOURCE manifest's directory, because the output "
            "manifest lives elsewhere and relative paths would no longer resolve."
        ),
    )
    parser.add_argument(
        "--max-unk",
        type=int,
        default=0,
        help="Keep entries with at most this many <unk> tokens (default 0 = drop any).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON path for the dropped-character report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("--output must differ from --manifest; this script never edits in place.")

    normalizer = TextNormalizer(
        preferred_language=args.language,
        zh_to_simplified=args.zh_to_simplified or None,
    )
    tokenizer = TextTokenizer(str(args.tokenizer), normalizer)
    unk_id = tokenizer.unk_token_id

    kept = dropped = 0
    missing_audio = 0
    offending: collections.Counter = collections.Counter()
    dropped_ids: list[str] = []
    unk_char_cache: dict[str, bool] = {}
    source_dir = args.manifest.resolve().parent

    def char_is_unk(ch: str) -> bool:
        """Does this single source character survive normalization + vocab lookup?"""
        cached = unk_char_cache.get(ch)
        if cached is None:
            folded = normalizer.to_simplified(ch) if normalizer.zh_to_simplified else ch
            cached = any(tokenizer.convert_tokens_to_ids(c)[0] == unk_id for c in folded)
            unk_char_cache[ch] = cached
        return cached

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(args.text_field, "")
            ids = tokenizer.encode(text, language=args.language)
            n_unk = sum(1 for i in ids if i == unk_id)

            if n_unk > args.max_unk:
                dropped += 1
                dropped_ids.append(record.get("id", ""))
                # Attribute the <unk> back to source characters for the report.
                for ch in text:
                    if char_is_unk(ch):
                        offending[ch] += 1
                continue

            if not args.keep_relative_audio:
                audio = record.get(args.audio_field)
                if audio:
                    resolved = Path(audio)
                    if not resolved.is_absolute():
                        resolved = (source_dir / resolved).resolve()
                    if not resolved.is_file():
                        missing_audio += 1
                    record[args.audio_field] = str(resolved)

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    total = kept + dropped
    print(f"[Done] kept {kept:,} / {total:,} ({kept / total * 100:.4f}%), dropped {dropped:,}")
    if missing_audio:
        print(
            f"[Warn] {missing_audio:,} kept records point at audio that does not exist "
            "— preprocessing will skip these."
        )
    if offending:
        print(f"[Info] characters causing <unk>: {len(offending)} distinct")
        print("       " + "  ".join(f"{c}({n})" for c, n in offending.most_common(40)))
    print(f"[Done] wrote {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "source": str(args.manifest),
                    "output": str(args.output),
                    "kept": kept,
                    "dropped": dropped,
                    "zh_to_simplified": bool(args.zh_to_simplified),
                    "offending_chars": offending.most_common(),
                    "dropped_ids": dropped_ids,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Done] wrote report {args.report}")


if __name__ == "__main__":
    main()
