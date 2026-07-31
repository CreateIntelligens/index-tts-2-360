"""
Local TTS service: JSON/multipart API plus a single-page UI for A/B-ing
finetuned checkpoints.

Text is expected in Mandarin Han characters. A model finetuned on the Taiwanese
corpus reads that text with Taiwanese pronunciation; you never type Taiwanese
Hanji on the input side.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from engine import (
    AUDIO_SUFFIXES,
    OUTPUTS_DIR,
    VOICES_DIR,
    discover_models,
    discover_tokenizers,
    discover_voices,
    engine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tts.api")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/app/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="IndexTTS2 local service", version="1.0")


@app.get("/api/health")
def health():
    return {"ok": True, **engine.status()}


@app.get("/api/models")
def models():
    return {
        "models": [entry.as_dict() for entry in discover_models()],
        "tokenizers": [entry.as_dict() for entry in discover_tokenizers()],
        "loaded": engine.loaded,
    }


@app.get("/api/voices")
def voices():
    return {"voices": [entry.as_dict() for entry in discover_voices()]}


@app.post("/api/voices")
async def add_voice(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Persist a reference clip so it can be reused from the dropdown."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(400, f"Unsupported audio type '{suffix}'. Allowed: {sorted(AUDIO_SUFFIXES)}")
    stem = Path(name or Path(file.filename or "voice").stem).stem
    target = VOICES_DIR / f"{stem}{suffix}"
    counter = 1
    while target.exists():
        target = VOICES_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    logger.info("saved reference voice %s", target)
    return {"saved": target.relative_to(VOICES_DIR).as_posix()}


@app.get("/api/voices/{voice_id:path}/audio")
def voice_audio(voice_id: str):
    """Stream a saved reference clip so the UI can preview it before synthesising."""
    target = (VOICES_DIR / voice_id).resolve()
    if VOICES_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "No such voice")
    suffix = target.suffix.lower()
    media = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")
    return FileResponse(target, media_type=media, filename=target.name)


@app.delete("/api/voices/{voice_id:path}")
def delete_voice(voice_id: str):
    target = (VOICES_DIR / voice_id).resolve()
    if VOICES_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "No such voice")
    target.unlink()
    return {"deleted": voice_id}


@app.post("/api/unload")
def unload():
    engine.unload()
    return {"ok": True, **engine.status()}


def _resolve_prompt(
    voice: Optional[str], upload: Optional[UploadFile], scratch: list[Path], required: bool
) -> Optional[Path]:
    """Prefer a per-request upload, else a saved voice id, else the first saved voice."""
    if upload is not None and upload.filename:
        suffix = Path(upload.filename).suffix.lower() or ".wav"
        if suffix not in AUDIO_SUFFIXES:
            raise HTTPException(400, f"Unsupported audio type '{suffix}'")
        target = UPLOAD_DIR / f"ref_{uuid.uuid4().hex}{suffix}"
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        scratch.append(target)
        return target
    if voice:
        candidate = (VOICES_DIR / voice).resolve()
        if VOICES_DIR.resolve() not in candidate.parents or not candidate.is_file():
            raise HTTPException(404, f"No such voice '{voice}'")
        return candidate
    if not required:
        return None
    available = discover_voices()
    if not available:
        raise HTTPException(
            400, "No reference audio: upload one with the request, or POST /api/voices first."
        )
    return available[0].path


@app.post("/api/tts")
async def tts(
    text: str = Form(...),
    model: Optional[str] = Form(None),
    tokenizer: Optional[str] = Form(None),
    # Speaker reference (timbre).
    voice: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
    # Emotion: 0 = same as speaker prompt, 1 = separate audio, 2 = 8-dim vector,
    # 3 = text description (experimental).
    emo_mode: int = Form(0),
    emo_voice: Optional[str] = Form(None),
    emo_reference: Optional[UploadFile] = File(None),
    emo_alpha: float = Form(0.5),
    emo_vector: Optional[str] = Form(None),
    emo_text: Optional[str] = Form(None),
    emo_random: bool = Form(False),
    # Generation.
    do_sample: bool = Form(True),
    temperature: float = Form(0.8),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    repetition_penalty: float = Form(10.0),
    num_beams: int = Form(3),
    length_penalty: float = Form(0.0),
    max_mel_tokens: int = Form(1500),
    max_text_tokens_per_sentence: int = Form(120),
    interval_silence: int = Form(200),
    seed: Optional[int] = Form(None),
):
    """
    Synthesise one utterance.

    `text` is Mandarin Han characters. A model finetuned on the Taiwanese corpus
    reads it with Taiwanese pronunciation; do not write Taiwanese Hanji.

    The speaker reference comes from an uploaded `reference` (this request only) or
    from `voice`, an id from GET /api/voices. Emotion mode 1 uses `emo_reference` /
    `emo_voice` the same way.
    """
    scratch: list[Path] = []
    try:
        prompt_path = _resolve_prompt(voice, reference, scratch, required=True)
        emo_audio_path = None
        if emo_mode == 1:
            emo_audio_path = _resolve_prompt(emo_voice, emo_reference, scratch, required=False)
            if emo_audio_path is None:
                raise HTTPException(400, "Emotion mode 1 needs emo_reference or emo_voice.")

        parsed_vector = None
        if emo_mode == 2:
            if not emo_vector:
                raise HTTPException(400, "Emotion mode 2 needs emo_vector.")
            try:
                parsed_vector = [float(x) for x in json.loads(emo_vector)]
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, f"emo_vector must be a JSON array of 8 numbers: {exc}")
            if len(parsed_vector) != 8:
                raise HTTPException(400, f"emo_vector needs 8 values, got {len(parsed_vector)}")

        try:
            # Synthesis is a long blocking GPU call. Running it directly inside an
            # async endpoint pins the event loop, so every other request — including
            # GET /api/audio for already-finished takes — stalls until it returns,
            # then they all complete at once. Hand it to a worker thread instead.
            result = await run_in_threadpool(
                engine.synthesize,
                text,
                model_id=model,
                tokenizer_id=tokenizer,
                prompt_path=prompt_path,
                emo_mode=emo_mode,
                emo_audio_path=emo_audio_path,
                emo_alpha=emo_alpha,
                emo_vector=parsed_vector,
                emo_text=emo_text or None,
                emo_random=emo_random,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                length_penalty=length_penalty,
                max_mel_tokens=max_mel_tokens,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                interval_silence=interval_silence,
                seed=seed,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # surface the real reason to the UI
            logger.exception("synthesis failed")
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

        return JSONResponse(
            {
                **result,
                "url": f"/api/audio/{result['file']}",
                "reference": prompt_path.name,
                "emo_reference": emo_audio_path.name if emo_audio_path else None,
            }
        )
    finally:
        for path in scratch:
            if path.exists():
                path.unlink()


@app.get("/api/audio/{name}")
def audio(name: str):
    target = (OUTPUTS_DIR / name).resolve()
    if OUTPUTS_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "No such audio")
    return FileResponse(target, media_type="audio/wav", filename=name)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
