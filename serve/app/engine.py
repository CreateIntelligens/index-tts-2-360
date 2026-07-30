"""
Model registry and inference wrapper for the local TTS service.

One GPU, one model in VRAM: switching checkpoints unloads the previous one, and a
lock serialises synthesis so concurrent requests queue instead of fighting over
the card.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tts.engine")

CHECKPOINTS = Path(os.environ.get("CHECKPOINTS_DIR", "/app/checkpoints"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
TOKENIZERS_DIR = Path(os.environ.get("TOKENIZERS_DIR", "/app/tokenizers"))
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/app/outputs"))

BASE_MODEL_ID = "base"
BASE_TOKENIZER_ID = "base"

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


@dataclass
class Entry:
    """A selectable checkpoint or tokenizer."""

    id: str
    label: str
    path: Path
    size_bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "path": str(self.path),
            "size_mb": round(self.size_bytes / 1024 / 1024, 1) if self.size_bytes else None,
            **self.extra,
        }


def _human(path: Path) -> str:
    return path.stem.replace("_", " ")


def discover_models() -> List[Entry]:
    """The stock checkpoint plus anything dropped into MODELS_DIR."""
    entries: List[Entry] = []
    base = CHECKPOINTS / "gpt.pth"
    if base.is_file():
        entries.append(
            Entry(
                id=BASE_MODEL_ID,
                label="base (IndexTTS-2 stock)",
                path=base,
                size_bytes=base.stat().st_size,
            )
        )
    if MODELS_DIR.is_dir():
        for path in sorted(MODELS_DIR.rglob("*.pth")):
            if not path.is_file():
                continue
            rel = path.relative_to(MODELS_DIR)
            entries.append(
                Entry(
                    id=rel.as_posix(),
                    label=_human(path),
                    path=path,
                    size_bytes=path.stat().st_size,
                )
            )
    return entries


def discover_tokenizers() -> List[Entry]:
    entries: List[Entry] = []
    base = CHECKPOINTS / "bpe.model"
    if base.is_file():
        entries.append(
            Entry(
                id=BASE_TOKENIZER_ID,
                label="base (bpe.model)",
                path=base,
                size_bytes=base.stat().st_size,
            )
        )
    if TOKENIZERS_DIR.is_dir():
        for path in sorted(TOKENIZERS_DIR.rglob("*.model")):
            if path.is_file():
                rel = path.relative_to(TOKENIZERS_DIR)
                entries.append(
                    Entry(id=rel.as_posix(), label=_human(path), path=path, size_bytes=path.stat().st_size)
                )
    return entries


def discover_voices() -> List[Entry]:
    entries: List[Entry] = []
    if VOICES_DIR.is_dir():
        for path in sorted(VOICES_DIR.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                rel = path.relative_to(VOICES_DIR)
                entries.append(
                    Entry(id=rel.as_posix(), label=rel.as_posix(), path=path, size_bytes=path.stat().st_size)
                )
    return entries


def _lookup(entries: List[Entry], wanted: Optional[str], kind: str) -> Entry:
    if not entries:
        raise FileNotFoundError(f"No {kind} available. Check the mounted directories.")
    if not wanted:
        return entries[0]
    for entry in entries:
        if entry.id == wanted:
            return entry
    raise FileNotFoundError(f"Unknown {kind} '{wanted}'.")


class TTSEngine:
    def __init__(self) -> None:
        self._tts = None
        self._loaded: tuple[str, str] | None = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # -- state -------------------------------------------------------------

    @property
    def loaded(self) -> Optional[Dict[str, str]]:
        if self._loaded is None:
            return None
        return {"model": self._loaded[0], "tokenizer": self._loaded[1]}

    def status(self) -> Dict[str, Any]:
        import torch

        info: Dict[str, Any] = {
            "loaded": self.loaded,
            "cuda": torch.cuda.is_available(),
            "zh_to_simplified": os.environ.get("INDEXTTS_ZH_T2S", "0") != "0",
            "last_error": self._load_error,
        }
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "free_mb": round(free / 1024 / 1024),
                "total_mb": round(total / 1024 / 1024),
            }
        return info

    # -- loading -----------------------------------------------------------

    def unload(self) -> None:
        import torch

        if self._tts is None:
            return
        logger.info("unloading %s", self._loaded)
        self._tts = None
        self._loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure(self, model_id: Optional[str], tokenizer_id: Optional[str]):
        model = _lookup(discover_models(), model_id, "model")
        tokenizer = _lookup(discover_tokenizers(), tokenizer_id, "tokenizer")
        key = (model.id, tokenizer.id)
        if self._tts is not None and self._loaded == key:
            return self._tts

        # Free the old weights before allocating the new ones; a single card
        # cannot hold two copies of a 1.5B model comfortably.
        self.unload()

        from indextts.infer_v2_modded import IndexTTS2

        started = time.time()
        logger.info("loading model=%s tokenizer=%s", model.id, tokenizer.id)
        try:
            self._tts = IndexTTS2(
                cfg_path=str(CHECKPOINTS / "config.yaml"),
                model_dir=str(CHECKPOINTS),
                gpt_checkpoint_path=str(model.path),
                bpe_model_path=str(tokenizer.path),
                use_fp16=os.environ.get("USE_FP16", "1") != "0",
                use_deepspeed=False,
            )
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._tts = None
            self._loaded = None
            raise
        self._loaded = key
        self._load_error = None
        logger.info("loaded in %.1fs", time.time() - started)
        return self._tts

    # -- synthesis ---------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        model_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        prompt_path: Path,
        emo_audio_path: Optional[Path] = None,
        emo_alpha: float = 1.0,
        emo_text: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 10.0,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        max_text_tokens_per_sentence: int = 120,
        interval_silence: int = 200,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("text is empty")

        with self._lock:  # one GPU: queue requests rather than interleave them
            tts = self.ensure(model_id, tokenizer_id)

            if seed is not None:
                import random

                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            out_path = OUTPUTS_DIR / f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
            started = time.time()
            tts.infer(
                spk_audio_prompt=str(prompt_path),
                text=text,
                output_path=str(out_path),
                emo_audio_prompt=str(emo_audio_path) if emo_audio_path else None,
                emo_alpha=emo_alpha,
                use_emo_text=bool(emo_text),
                emo_text=emo_text or None,
                interval_silence=interval_silence,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                verbose=False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                length_penalty=length_penalty,
                max_mel_tokens=max_mel_tokens,
            )
            elapsed = time.time() - started

        if not out_path.is_file():
            raise RuntimeError("inference produced no audio file")

        return {
            "file": out_path.name,
            "path": str(out_path),
            "seconds": round(elapsed, 2),
            "model": (self._loaded or ("?", "?"))[0],
            "tokenizer": (self._loaded or ("?", "?"))[1],
        }


engine = TTSEngine()
