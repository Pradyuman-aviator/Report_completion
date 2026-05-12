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
    title="MedFill API v1.0.0",
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
# Routes — Frontend API  (called by Fronted_medfill React app)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/upload-scan", tags=["Frontend API"])
async def upload_scan(
    file: UploadFile = File(..., description="Medical report image (JPEG/PNG)"),
) -> JSONResponse:
    """
    Step 1: Upload a report image → EasyOCR → regex extraction.

    Returns extracted features + patient info for the frontend to display.
    This does NOT run the ML imputation model yet.
    """
    _validate_extension(file.filename or "")
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="medfill_scan_") as tmp_dir:
        image_path = _save_upload(file, tmp_dir)
        logger.info(f"[Gateway] /api/v1/upload-scan  file={file.filename}  size={image_path.stat().st_size}B")

        try:
            from ai_agents.easyocr_agent import EasyOCRAgent
            from infer import direct_parse_ocr, UNIFIED
            import re

            # Run EasyOCR
            agent      = EasyOCRAgent(use_gpu=True)
            ocr_result = agent.process(str(image_path))
            raw_ocr    = ocr_result.raw_text

            # Extract features via regex
            known = direct_parse_ocr(raw_ocr)

            # Extract patient name
            pt_name = "Unknown"
            m = re.search(r"MR[\.:]?\s+([A-Z][A-Z\s]+)", raw_ocr)
            if m:
                pt_name = "MR. " + m.group(1).strip()[:30]
            else:
                m2 = re.search(r"(?:patient|name)\s*[:\-]\s*(.+)", raw_ocr, re.IGNORECASE)
                if m2:
                    pt_name = m2.group(1).strip()[:40]

            # Extract report date
            report_date = None
            m_date = re.search(r"(\d{1,2}[-/]\w{3,9}[-/]\d{2,4})", raw_ocr)
            if m_date:
                report_date = m_date.group(1)

            n_extracted = sum(1 for v in known.values() if v is not None)

        except Exception as e:
            logger.exception(f"[Gateway] upload-scan error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

        elapsed = time.perf_counter() - t_start
        logger.info(f"[Gateway] /api/v1/upload-scan  done in {elapsed:.1f}s  extracted={n_extracted}/25")

        return JSONResponse(content={
            "ocr_confidence":  round(ocr_result.confidence, 3),
            "ocr_elapsed_s":   round(elapsed, 2),
            "patient_name":    pt_name,
            "report_date":     report_date,
            "report_type":     "blood_test",
            "raw_ocr_text":    raw_ocr,
            "extracted_features": {k: v for k, v in known.items()},
            "n_extracted":     n_extracted,
            "n_total":         25,
        })


@app.post("/api/v1/predict-disease", tags=["Frontend API"])
async def predict_disease(body: Dict[str, Any]) -> JSONResponse:
    """
    Step 2: Run ML imputation on the extracted features.

    Accepts the JSON returned by /api/v1/upload-scan.
    Returns the complete 25-feature panel with extracted/predicted tags,
    formatted for the frontend PredictionPanel component.
    """
    t_start = time.perf_counter()

    try:
        from infer import impute, UNIFIED, direct_parse_ocr

        # Accept both raw_ocr_text and pre-extracted features
        known_raw: Dict = body.get("extracted_features") or {}
        raw_ocr:   str  = body.get("raw_ocr_text", "")

        # If features missing but we have raw OCR, re-parse
        if not any(v is not None for v in known_raw.values()) and raw_ocr:
            known_raw = direct_parse_ocr(raw_ocr)

        # Convert all values to float or None
        known = {}
        for k in UNIFIED:
            v = known_raw.get(k)
            if v is None:
                known[k] = None
            else:
                try:
                    known[k] = float(v)
                except (TypeError, ValueError):
                    known[k] = None

        # Run imputation
        complete = impute(known)

        # Build frontend-friendly panel
        # FEATURE_META for display labels/units
        FEATURE_META = {
            "Hemoglobin": {"label": "Hemoglobin",         "unit": "g/dL",   "normal": [10.0, 18.0]},
            "MCH":        {"label": "MCH",                "unit": "pg",     "normal": [25.0, 35.0]},
            "MCHC":       {"label": "MCHC",               "unit": "g/dL",   "normal": [30.0, 38.0]},
            "MCV":        {"label": "MCV",                "unit": "fL",     "normal": [60.0, 100.0]},
            "Gender":     {"label": "Gender",             "unit": None,     "normal": None},
            "age":        {"label": "Age",                "unit": "yrs",    "normal": [1, 120]},
            "bp":         {"label": "Blood Pressure",     "unit": "mmHg",   "normal": [60, 140]},
            "bgr":        {"label": "Blood Glucose",      "unit": "mg/dL",  "normal": [70, 200]},
            "bu":         {"label": "Blood Urea",         "unit": "mg/dL",  "normal": [5, 60]},
            "sc":         {"label": "Serum Creatinine",   "unit": "mg/dL",  "normal": [0.4, 1.4]},
            "sod":        {"label": "Sodium",             "unit": "mEq/L",  "normal": [135, 145]},
            "pot":        {"label": "Potassium",          "unit": "mEq/L",  "normal": [3.5, 5.0]},
            "pcv":        {"label": "Packed Cell Vol.",   "unit": "%",      "normal": [35, 55]},
            "wc":         {"label": "WBC Count",          "unit": "/µL",    "normal": [4000, 11000]},
            "rc":         {"label": "RBC Count",          "unit": "M/µL",   "normal": [3.5, 5.5]},
            "rbc":        {"label": "RBC Morphology",     "unit": None,     "normal": None},
            "pc":         {"label": "Pus Cells",          "unit": None,     "normal": None},
            "pcc":        {"label": "Pus Cell Clumps",    "unit": None,     "normal": None},
            "ba":         {"label": "Bacteria",           "unit": None,     "normal": None},
            "htn":        {"label": "Hypertension",       "unit": None,     "normal": None},
            "dm":         {"label": "Diabetes Mellitus",  "unit": None,     "normal": None},
            "cad":        {"label": "Coronary Artery",    "unit": None,     "normal": None},
            "appet":      {"label": "Appetite",           "unit": None,     "normal": None},
            "pe":         {"label": "Pedal Edema",        "unit": None,     "normal": None},
            "ane":        {"label": "Anaemia",            "unit": None,     "normal": None},
        }

        panel = {}
        for col, result in complete.items():
            was_extracted = known.get(col) is not None
            meta          = FEATURE_META.get(col, {"label": col, "unit": None, "normal": None})

            # Compute human-readable display value for categorical features
            raw_val = result["value"]
            if col == "Gender":
                display = "Male" if raw_val >= 0.5 else "Female"
            elif col in ("rbc", "pc"):
                display = "Normal" if raw_val >= 0.5 else "Abnormal"
            elif col in ("pcc", "ba"):
                display = "Present" if raw_val >= 0.5 else "Not Present"
            elif col == "appet":
                display = "Good" if raw_val >= 0.5 else "Poor"
            elif col in ("htn", "dm", "cad", "pe", "ane"):
                display = "Yes" if raw_val >= 0.5 else "No"
            else:
                display = None   # numeric — frontend formats it

            panel[col] = {
                "label":         meta["label"],
                "value":         raw_val,
                "display_value": display,
                "unit":          meta["unit"],
                "predicted":     not was_extracted,
                "source":        "extracted" if was_extracted else "predicted",
                "note":          result.get("note", ""),
            }

    except Exception as e:
        logger.exception(f"[Gateway] predict-disease error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    elapsed = time.perf_counter() - t_start
    n_pred  = sum(1 for v in panel.values() if v["predicted"])
    logger.info(f"[Gateway] /api/v1/predict-disease  done in {elapsed:.1f}s  predicted={n_pred}/25")

    return JSONResponse(content={"complete_panel": panel, "elapsed_s": round(elapsed, 2)})



# ──────────────────────────────────────────────────────────────────────────────
# Routes — Combined analyze endpoint  (one call: OCR + imputation)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/analyze", tags=["Frontend API"])
async def analyze(
    file: UploadFile = File(..., description="Medical report image (JPEG/PNG)"),
) -> JSONResponse:
    """
    All-in-one endpoint for the React frontend.
    Runs: EasyOCR → regex extraction → ML imputation.
    Returns a single JSON with patient info, lab panels, and complete predicted panel.
    """
    _validate_extension(file.filename or "")
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="medfill_analyze_") as tmp_dir:
        image_path = _save_upload(file, tmp_dir)
        logger.info(f"[Gateway] /api/v1/analyze  file={file.filename}")

        try:
            import re as _re
            from ai_agents.easyocr_agent import EasyOCRAgent
            from infer import direct_parse_ocr, impute, UNIFIED

            # ── Step 1: OCR ────────────────────────────────────────────────
            agent      = EasyOCRAgent(use_gpu=True)
            ocr_result = agent.process(str(image_path))
            raw_ocr    = ocr_result.raw_text

            # ── Step 2: Regex extraction ───────────────────────────────────
            known = direct_parse_ocr(raw_ocr)

            # ── Step 3: ML Imputation ──────────────────────────────────────
            complete = impute(known)

            # ── Step 4: Build patient info from OCR text ──────────────────
            pt_name, pt_age, pt_sex, pt_id, report_date, facility_name = (
                None, None, None, None, None, None
            )
            # Patient name: "MR. RAHUL SHARMA"
            m = _re.search(r"MR[\.:]?\s+([A-Z][A-Z\s]+)", raw_ocr)
            if m:
                pt_name = "MR. " + m.group(1).strip()[:30]

            # Age: "52 Years"
            m = _re.search(r"(\d{1,3})\s*[Yy]ears?", raw_ocr)
            if m:
                pt_age = int(m.group(1))
            elif known.get("age"):
                pt_age = int(known["age"])

            # Sex: "Male" / "Female"
            m = _re.search(r"\b(Male|Female)\b", raw_ocr, _re.IGNORECASE)
            if m:
                pt_sex = m.group(1).capitalize()
            elif known.get("Gender") is not None:
                pt_sex = "Male" if known["Gender"] >= 0.5 else "Female"

            # Patient ID
            m = _re.search(r"(?:Patient ID|UHID|MRN)\s*[:/]?\s*([A-Z0-9\-]+)", raw_ocr, _re.IGNORECASE)
            if m:
                pt_id = m.group(1).strip()

            # Report date
            m = _re.search(r"(\d{1,2}[-/][A-Za-z]+[-/]\d{2,4}|\d{4}-\d{2}-\d{2})", raw_ocr)
            if m:
                report_date = m.group(1)

            # Facility
            if "apollo" in raw_ocr.lower():
                facility_name = "Apollo Diagnostics"
            elif "thyrocare" in raw_ocr.lower():
                facility_name = "Thyrocare"
            else:
                m = _re.search(r"^([A-Z][A-Za-z\s&]+(?:Lab|Hospital|Diagnostics|Clinic|Centre|Center))", raw_ocr, _re.MULTILINE)
                if m:
                    facility_name = m.group(1).strip()

            # ── Step 5: Build lab_panels from extracted values ─────────────
            # These are the ACTUAL values from the report (not imputed)
            LAB_PANEL_ROWS = [
                # (title, feature_key, unit, ref_low, ref_high)
                ("Hemoglobin (Hb)",             "Hemoglobin", "g/dL",   13.0, 17.0),
                ("Red Blood Cell Count",         "rc",         "M/µL",   4.5,  5.5),
                ("Packed Cell Volume (PCV)",     "pcv",        "%",      40.0, 50.0),
                ("White Blood Cell Count (TLC)", "wc",         "/µL",    4000, 11000),
                ("Mean Corpuscular Hb (MCH)",    "MCH",        "pg",     27.0, 32.0),
                ("Mean Corpuscular Volume (MCV)","MCV",        "fL",     80.0, 100.0),
                ("Mean Corp. Hb Conc (MCHC)",   "MCHC",       "g/dL",   32.0, 36.0),
                ("Blood Urea",                   "bu",         "mg/dL",  15.0, 40.0),
                ("Serum Creatinine",             "sc",         "mg/dL",  0.6,  1.2),
                ("Sodium (Na+)",                 "sod",        "mEq/L",  135,  145),
                ("Potassium (K+)",               "pot",        "mEq/L",  3.5,  5.0),
                ("Blood Glucose (Random)",       "bgr",        "mg/dL",  None, 140),
            ]

            cbc_rows, rft_rows = [], []
            for test_name, feat, unit, ref_lo, ref_hi in LAB_PANEL_ROWS:
                val = known.get(feat)
                if val is None:
                    # Use imputed value
                    val = complete.get(feat, {}).get("value")
                    source = "pending"
                else:
                    source = "measured"

                if val is not None:
                    # Determine flag
                    flag = "unknown"
                    if ref_lo is not None and ref_hi is not None:
                        if val < ref_lo:   flag = "low"
                        elif val > ref_hi: flag = "high"
                        else:              flag = "normal"
                    elif ref_hi is not None and val > ref_hi:
                        flag = "high"

                    ref_text = f"{ref_lo} – {ref_hi}" if ref_lo and ref_hi else (f"< {ref_hi}" if ref_hi else "")
                    row = {
                        "test_name":      test_name,
                        "value":          round(float(val), 2),
                        "value_text":     str(round(float(val), 2)),
                        "unit":           unit,
                        "reference_text": ref_text,
                        "reference_low":  ref_lo,
                        "reference_high": ref_hi,
                        "flag":           flag,
                        "source":         source,
                    }
                    if feat in ("Hemoglobin", "rc", "pcv", "wc", "MCH", "MCV", "MCHC"):
                        cbc_rows.append(row)
                    else:
                        rft_rows.append(row)

            lab_panels = []
            if cbc_rows:
                lab_panels.append({"panel_name": "Complete Blood Count (CBC)", "results": cbc_rows})
            if rft_rows:
                lab_panels.append({"panel_name": "Renal & Biochemistry Panel", "results": rft_rows})

            # ── Step 6: Clinical history ───────────────────────────────────
            htn_val = known.get("htn")
            dm_val  = known.get("dm")
            cad_val = known.get("cad")
            bp_val  = known.get("bp")

            # ── Step 7: Build complete_panel for ML result display ─────────
            FEATURE_META = {
                "Hemoglobin": {"label": "Hemoglobin",        "unit": "g/dL",  "normal": [10.0, 18.0]},
                "MCH":        {"label": "MCH",               "unit": "pg",    "normal": [25.0, 35.0]},
                "MCHC":       {"label": "MCHC",              "unit": "g/dL",  "normal": [30.0, 38.0]},
                "MCV":        {"label": "MCV",               "unit": "fL",    "normal": [60.0, 100.0]},
                "Gender":     {"label": "Gender",            "unit": None,    "normal": None},
                "age":        {"label": "Age",               "unit": "yrs",   "normal": [1, 120]},
                "bp":         {"label": "Blood Pressure",    "unit": "mmHg",  "normal": [60, 140]},
                "bgr":        {"label": "Blood Glucose",     "unit": "mg/dL", "normal": [70, 200]},
                "bu":         {"label": "Blood Urea",        "unit": "mg/dL", "normal": [5, 60]},
                "sc":         {"label": "Serum Creatinine",  "unit": "mg/dL", "normal": [0.4, 1.4]},
                "sod":        {"label": "Sodium",            "unit": "mEq/L", "normal": [135, 145]},
                "pot":        {"label": "Potassium",         "unit": "mEq/L", "normal": [3.5, 5.0]},
                "pcv":        {"label": "Packed Cell Vol.",  "unit": "%",     "normal": [35, 55]},
                "wc":         {"label": "WBC Count",         "unit": "/µL",   "normal": [4000, 11000]},
                "rc":         {"label": "RBC Count",         "unit": "M/µL",  "normal": [3.5, 5.5]},
                "rbc":        {"label": "RBC Morphology",    "unit": None,    "normal": None},
                "pc":         {"label": "Pus Cells",         "unit": None,    "normal": None},
                "pcc":        {"label": "Pus Cell Clumps",   "unit": None,    "normal": None},
                "ba":         {"label": "Bacteria",          "unit": None,    "normal": None},
                "htn":        {"label": "Hypertension",      "unit": None,    "normal": None},
                "dm":         {"label": "Diabetes Mellitus", "unit": None,    "normal": None},
                "cad":        {"label": "Coronary Artery",   "unit": None,    "normal": None},
                "appet":      {"label": "Appetite",          "unit": None,    "normal": None},
                "pe":         {"label": "Pedal Edema",       "unit": None,    "normal": None},
                "ane":        {"label": "Anaemia",           "unit": None,    "normal": None},
            }

            complete_panel = {}
            for col, result in complete.items():
                was_extracted = known.get(col) is not None
                meta = FEATURE_META.get(col, {"label": col, "unit": None, "normal": None})
                raw_val = result["value"]

                # Human-readable display value for categoricals
                if col == "Gender":    display = "Male" if raw_val >= 0.5 else "Female"
                elif col in ("rbc", "pc"):   display = "Normal" if raw_val >= 0.5 else "Abnormal"
                elif col in ("pcc", "ba"):   display = "Present" if raw_val >= 0.5 else "Not Present"
                elif col == "appet":         display = "Good" if raw_val >= 0.5 else "Poor"
                elif col in ("htn", "dm", "cad", "pe", "ane"): display = "Yes" if raw_val >= 0.5 else "No"
                else: display = None

                complete_panel[col] = {
                    "label":         meta["label"],
                    "value":         round(raw_val, 3),
                    "display_value": display,
                    "unit":          meta["unit"],
                    "predicted":     not was_extracted,
                    "source":        "extracted" if was_extracted else "predicted",
                }

        except Exception as e:
            logger.exception(f"[Gateway] analyze error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

        elapsed     = time.perf_counter() - t_start
        n_extracted = sum(1 for v in known.values() if v is not None)
        n_predicted = 25 - n_extracted
        logger.info(f"[Gateway] /api/v1/analyze  done in {elapsed:.1f}s  extracted={n_extracted}  predicted={n_predicted}")

        return JSONResponse(content={
            # Patient section — feeds PatientDashboard
            "patient": {
                "name":       pt_name,
                "patient_id": pt_id,
                "age_years":  pt_age,
                "sex":        pt_sex,
                "date_of_birth": None,
                "contact_number": None,
                "address":    None,
            },
            "facility": {
                "name": facility_name,
            } if facility_name else None,
            "vitals": {
                "blood_pressure_systolic":  bp_val,
                "blood_pressure_diastolic": None,
                "heart_rate_bpm":           None,
                "temperature_celsius":      None,
                "spo2_percent":             None,
                "weight_kg":                None,
                "height_cm":                None,
            } if bp_val else None,
            "clinical_history": {
                "hypertension":     "Yes" if htn_val and htn_val >= 0.5 else ("No" if htn_val is not None else None),
                "diabetes":         "Yes" if dm_val  and dm_val  >= 0.5 else ("No" if dm_val  is not None else None),
                "coronary_artery":  "Yes" if cad_val and cad_val >= 0.5 else ("No" if cad_val is not None else None),
            },
            "lab_panels":     lab_panels,
            "diagnoses":      [],
            "medications":    [],
            # ML Panel — feeds PredictionPanel / ReportAnalysis
            "complete_panel": complete_panel,
            # Meta
            "ocr_confidence": round(ocr_result.confidence, 3),
            "elapsed_s":      round(elapsed, 2),
            "n_extracted":    n_extracted,
            "n_predicted":    n_predicted,
            "report_type":    "blood_test",
            "report_date":    report_date,
        })


# ──────────────────────────────────────────────────────────────────────────────
# Dev runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api_gateway.main:app", host="0.0.0.0", port=8000, reload=True)
