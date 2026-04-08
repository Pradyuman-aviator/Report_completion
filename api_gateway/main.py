"""
api_gateway/main.py
====================
FastAPI HTTP Gateway for the MedFill Medical Report Pipeline
------------------------------------------------------------

Endpoints:
    POST /process          — Submit a single medical report image → JSON payload
    POST /process/batch    — Submit multiple images → list of JSON payloads
    GET  /health           — Liveness + Ollama connectivity check
    GET  /models           — List available Ollama models on this machine

Run locally:
    uvicorn api_gateway.main:app --reload --port 8000

Example:
    curl -X POST http://localhost:8000/process -F "file=@report.jpg"
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# App & Middleware
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedFill API Gateway",
    description=(
        "Local-first medical report OCR and structuring pipeline.\n"
        "All inference runs entirely on-device via Ollama — zero cloud calls."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_HOST        = "http://localhost:11434"
VISION_MODEL       = "llava:7b"
STRUCTURING_MODEL  = "llama3"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )


def _save_upload(upload: UploadFile, tmp_dir: str) -> Path:
    dest = Path(tmp_dir) / (upload.filename or "upload")
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# Routes — System
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> Dict[str, Any]:
    """Liveness + readiness check. Returns whether Ollama server is reachable."""
    info: Dict[str, Any] = {"status": "ok", "ollama": False, "models": []}
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
        result = client.list()
        info["ollama"] = True
        info["models"] = [m["name"] for m in result.get("models", [])]
    except Exception as e:
        info["status"]       = "degraded"
        info["ollama_error"] = str(e)
    return info


@app.get("/models", tags=["System"])
def list_models(
    ollama_host: str = Query(default=OLLAMA_HOST, description="Ollama base URL"),
) -> Dict[str, Any]:
    """List all Ollama models currently pulled on this machine."""
    try:
        import ollama
        client = ollama.Client(host=ollama_host)
        result = client.list()
        models = [m["name"] for m in result.get("models", [])]
        return {"ollama_host": ollama_host, "models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot reach Ollama at {ollama_host}: {e}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Routes — Pipeline
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/process", tags=["Pipeline"])
async def process_single(
    file:              UploadFile = File(..., description="Medical report image (JPEG/PNG)"),
    vision_model:      str  = Query(default=VISION_MODEL,      description="Ollama VLM tag"),
    structuring_model: str  = Query(default=STRUCTURING_MODEL, description="Ollama LLM tag"),
    ollama_host:       str  = Query(default=OLLAMA_HOST,       description="Ollama base URL"),
    skip_preprocess:   bool = Query(default=False,             description="Skip OpenCV preprocessing"),
    save_cleaned:      bool = Query(default=False,             description="Persist cleaned images to disk"),
) -> JSONResponse:
    """
    Process a single medical report image through the full pipeline.

    Runs: **OpenCV preprocessing → VLM OCR → JSON structuring → Pydantic validation**

    Requires `ollama serve` with `llama3.2-vision` and `llama3` pulled.
    """
    _validate_extension(file.filename or "")
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="medfill_") as tmp_dir:
        image_path = _save_upload(file, tmp_dir)
        logger.info(f"[Gateway] /process  file={file.filename}  size={image_path.stat().st_size}B")

        try:
            from ai_agents.pipeline import MedicalReportPipeline
            pipeline = MedicalReportPipeline(
                vision_model              = vision_model,
                structuring_model         = structuring_model,
                ollama_host               = ollama_host,
                save_cleaned_images       = save_cleaned,
                skip_vision_preprocessing = skip_preprocess,
            )
            payload = pipeline.process(image_path)
        except ConnectionError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except Exception as e:
            logger.exception(f"[Gateway] Pipeline error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline failed: {e}",
            )

        elapsed = time.perf_counter() - t_start
        logger.info(f"[Gateway] /process  done in {elapsed:.1f}s")
        return JSONResponse(
            content=payload.model_dump(mode="json"),
            headers={"X-Elapsed-Seconds": f"{elapsed:.2f}"},
        )


@app.post("/process/batch", tags=["Pipeline"])
async def process_batch(
    files:             List[UploadFile] = File(..., description="One or more medical report images"),
    vision_model:      str  = Query(default=VISION_MODEL),
    structuring_model: str  = Query(default=STRUCTURING_MODEL),
    ollama_host:       str  = Query(default=OLLAMA_HOST),
    fail_fast:         bool = Query(default=False, description="Stop on first error"),
) -> JSONResponse:
    """
    Process multiple medical report images in a single request.

    Returns a list of payloads (null for failed images unless `fail_fast=true`).
    """
    for f in files:
        _validate_extension(f.filename or "")

    t_start = time.perf_counter()
    results: List[Optional[Dict[str, Any]]] = []
    errors:  List[Dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="medfill_batch_") as tmp_dir:
        saved_paths = [_save_upload(f, tmp_dir) for f in files]
        logger.info(f"[Gateway] /process/batch  count={len(saved_paths)}")

        try:
            from ai_agents.pipeline import MedicalReportPipeline
            pipeline = MedicalReportPipeline(
                vision_model      = vision_model,
                structuring_model = structuring_model,
                ollama_host       = ollama_host,
                save_cleaned_images=False,
            )
        except ConnectionError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

        for path in saved_paths:
            try:
                payload = pipeline.process(path)
                results.append(payload.model_dump(mode="json"))
            except Exception as e:
                logger.error(f"[Gateway] Batch error for {path.name}: {e}")
                errors.append({"file": path.name, "error": str(e)})
                results.append(None)
                if fail_fast:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Batch aborted at {path.name}: {e}",
                    )

    elapsed = time.perf_counter() - t_start
    return JSONResponse(
        content={
            "total":    len(files),
            "success":  sum(1 for r in results if r is not None),
            "failures": len(errors),
            "results":  results,
            "errors":   errors,
        },
        headers={"X-Elapsed-Seconds": f"{elapsed:.2f}"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dev runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api_gateway.main:app", host="0.0.0.0", port=8000, reload=True)
