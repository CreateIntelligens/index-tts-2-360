"""
Local TTS service: JSON/multipart API plus a single-page UI for A/B-ing
finetuned checkpoints.

Text is expected in Mandarin Han characters. A model finetuned on the Taiwanese
corpus reads that text with Taiwanese pronunciation; you never type Taiwanese
Hanji on the input side.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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


@app.post("/api/tts")
async def tts(
    text: str = Form(...),
    model: Optional[str] = Form(None),
    tokenizer: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
    emo_text: Optional[str] = Form(None),
    emo_alpha: float = Form(1.0),
    temperature: float = Form(0.8),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    repetition_penalty: float = Form(10.0),
    num_beams: int = Form(3),
    max_mel_tokens: int = Form(1500),
    max_text_tokens_per_sentence: int = Form(120),
    interval_silence: int = Form(200),
    seed: Optional[int] = Form(None),
):
    """
    Synthesise one utterance.

    The voice reference comes either from an uploaded `reference` file (used for
    this request only) or from `voice`, an id returned by GET /api/voices.
    """
    temp_upload: Optional[Path] = None
    try:
        if reference is not None and reference.filename:
            suffix = Path(reference.filename).suffix.lower() or ".wav"
            if suffix not in AUDIO_SUFFIXES:
                raise HTTPException(400, f"Unsupported reference audio type '{suffix}'")
            temp_upload = UPLOAD_DIR / f"ref_{uuid.uuid4().hex}{suffix}"
            with temp_upload.open("wb") as handle:
                shutil.copyfileobj(reference.file, handle)
            prompt_path = temp_upload
        elif voice:
            candidate = (VOICES_DIR / voice).resolve()
            if VOICES_DIR.resolve() not in candidate.parents or not candidate.is_file():
                raise HTTPException(404, f"No such voice '{voice}'")
            prompt_path = candidate
        else:
            available = discover_voices()
            if not available:
                raise HTTPException(
                    400,
                    "No reference audio: upload one with the request, or POST /api/voices first.",
                )
            prompt_path = available[0].path

        try:
            result = engine.synthesize(
                text,
                model_id=model,
                tokenizer_id=tokenizer,
                prompt_path=prompt_path,
                emo_text=emo_text or None,
                emo_alpha=emo_alpha,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
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
            }
        )
    finally:
        if temp_upload is not None and temp_upload.exists():
            temp_upload.unlink()


@app.get("/api/audio/{name}")
def audio(name: str):
    target = (OUTPUTS_DIR / name).resolve()
    if OUTPUTS_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "No such audio")
    return FileResponse(target, media_type="audio/wav", filename=name)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
