"""
ai_agents/structuring_agent.py
================================
Structuring Agent — OCR Text → Validated MedicalReportPayload
--------------------------------------------------------------

Receives the raw OCR text string from VisionAgent and uses a locally
running Ollama LLM (Llama-3-8B) to parse it into a structured JSON object
that is then validated against our Pydantic MedicalReportPayload schema.

Architecture decisions implemented:
  1. Single dominant report  — extracts one MedicalReportPayload, flags
                               any secondary document content in
                               metadata.extraction_warnings.
  2. Prompt engineering + Pydantic retry loop  — no instructor/tool-calling
                               libraries. The JSON schema is injected into the
                               prompt. Up to MAX_RETRIES=3 attempts; on each
                               ValidationError the exact Pydantic error
                               messages are fed back to the LLM for correction.
  3. Soft-fail with partial payload  — if all retries are exhausted the agent
                               returns the best parseable subset of fields with
                               confidence=0.0 rather than raising an exception.
                               Downstream V-JEPA handles the missing values.

Ollama setup:
    ollama pull llama3          (or llama3:8b / llama3.1:8b)
    ollama serve

Usage:
    >>> from ai_agents.structuring_agent import StructuringAgent
    >>> agent = StructuringAgent()
    >>> payload = agent.structure(raw_ocr_text, ocr_confidence=0.85)
    >>> print(payload.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

import ollama
from pydantic import ValidationError

from ai_agents.schemas import (
    AbnormalityFlag,
    ExtractionMetadata,
    MedicalReportPayload,
    ReportType,
    Sex,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_RETRIES       = 3
OLLAMA_TIMEOUT_S  = 120    # Llama-3-8B on M-series takes ~20-60 s first call
DEFAULT_MODEL     = "llama3"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Prompt Construction
# ──────────────────────────────────────────────────────────────────────────────

def _build_schema_excerpt() -> str:
    """
    Return a compact, LLM-readable summary of the MedicalReportPayload schema.

    We do NOT dump the full verbose JSON Schema (too many tokens).
    Instead we provide a handcrafted field-by-field description that fits
    within the 8B model's effective context window on a quantised setup.
    """
    return """
{
  "report_type": "one of: blood_test|urine_test|imaging_report|pathology|prescription|discharge_summary|consultation_note|ecg_report|vaccination_record|unknown",
  "report_date": "YYYY-MM-DD or null",
  "report_id": "string or null",
  "patient": {
    "name": "string or null",
    "patient_id": "string or null",
    "date_of_birth": "YYYY-MM-DD or null",
    "age_years": "number or null",
    "sex": "male|female|other|unknown",
    "contact_number": "string or null",
    "address": "string or null"
  },
  "facility": {
    "name": "string or null",
    "address": "string or null",
    "phone": "string or null",
    "accreditation": "string or null"
  },
  "referring_doctor": { "name": "string or null", "specialisation": "string or null", "registration_no": "string or null", "signature_present": false },
  "reporting_doctor": { "name": "string or null", "specialisation": "string or null", "registration_no": "string or null", "signature_present": false },
  "vitals": {
    "blood_pressure_systolic": "number or null",
    "blood_pressure_diastolic": "number or null",
    "heart_rate_bpm": "number or null",
    "temperature_celsius": "number or null",
    "spo2_percent": "number or null",
    "weight_kg": "number or null",
    "height_cm": "number or null"
  },
  "lab_panels": [
    {
      "panel_name": "e.g. CBC, LFT, KFT, Lipid Profile",
      "results": [
        {
          "test_name": "exact name as printed",
          "value": "numeric value or null",
          "value_text": "raw string if non-numeric e.g. Positive/Negative",
          "unit": "string or null",
          "reference_low": "number or null",
          "reference_high": "number or null",
          "reference_text": "full range string or null",
          "flag": "normal|high|low|critical|unknown",
          "method": "string or null"
        }
      ]
    }
  ],
  "diagnoses": [
    { "name": "string", "icd10_code": "string or null", "severity": "mild|moderate|severe|unknown", "is_primary": true, "notes": "string or null" }
  ],
  "medications": [
    { "drug_name": "string", "dosage": "string or null", "frequency": "string or null", "duration": "string or null", "route": "oral|intravenous|topical|inhaled|subcutaneous|other", "instructions": "string or null" }
  ],
  "imaging": {
    "modality": "X-Ray|USG|MRI|CT|PET|null",
    "region": "string or null",
    "findings": "full findings text or null",
    "impression": "radiologist impression or null"
  },
  "clinical_summary": "free text or null",
  "follow_up_instructions": "free text or null"
}""".strip()


def _focus_ocr_text(raw_ocr_text: str, max_chars: int = 3500) -> str:
    """
    Filter OCR text to keep only lines most likely to contain lab results.
    Reduces 9000+ char OCR dumps to ~2000 chars that llama3 can reliably parse.
    """
    import re
    lines = raw_ocr_text.splitlines()
    keep = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Always keep: lines with digits (test values), patient info, section headers
        has_number  = bool(re.search(r'\d', stripped))
        is_header   = any(kw in stripped.lower() for kw in [
            "patient", "name", "age", "sex", "gender", "date", "report",
            "blood", "urine", "renal", "count", "cbc", "kft", "rft", "lft",
            "panel", "test", "result", "reference", "hemoglobin", "haemoglobin",
            "creatinine", "urea", "sodium", "potassium", "glucose", "history",
            "pressure", "hypertension", "diabetes", "coronary",
        ])
        if has_number or is_header:
            keep.append(stripped)

    focused = "\n".join(keep)
    # Hard cap to keep within llama3 context
    if len(focused) > max_chars:
        focused = focused[:max_chars] + "\n...[truncated]"
    return focused


def _build_initial_prompt(raw_ocr_text: str, schema_str: str) -> str:
    return f"""You are a precise medical data extraction AI.

Your task is to parse the following OCR-extracted text from a scanned medical report and output a single, valid JSON object that matches the schema below.

STRICT RULES:
1. Output ONLY the JSON object. No markdown, no code fences, no explanation before or after.
2. Every key in the schema must be present. Use null for missing or illegible values.
3. Numbers must be actual JSON numbers (not strings). Dates must be "YYYY-MM-DD".
4. CRITICAL: For lab_panels — you MUST extract ALL test results from the text into lab_panels. This is the most important field. Do NOT leave lab_panels as an empty list [] if test results appear in the text. Group CBC tests in one panel, RFT/KFT in another, etc.
5. For PENDING or '---' values: set value=null and value_text="PENDING".
6. For clinical history fields like Hypertension/Diabetes, extract them as lab results under a panel called "Clinical History".
7. Do NOT invent or hallucinate values not present in the text. Use null instead.

SCHEMA TO FOLLOW:
{schema_str}

OCR TEXT TO PARSE:
---
{raw_ocr_text}
---

Output the JSON object now (remember: lab_panels MUST contain the test results):"""


def _build_retry_prompt(
    raw_ocr_text: str,
    schema_str:   str,
    previous_json: str,
    validation_errors: str,
    attempt: int,
) -> str:
    return f"""Your previous JSON output (attempt {attempt}) failed validation.

VALIDATION ERRORS:
{validation_errors}

YOUR PREVIOUS (INVALID) JSON:
{previous_json}

Your task is to fix ONLY the errors listed above and output a corrected, complete JSON object.
Keep all correctly parsed fields from your previous attempt unchanged.

STRICT RULES (same as before):
- Output ONLY the valid JSON object. No explanation, no markdown, no code fences.
- Use null for truly missing values.
- Numbers must be JSON numbers, not strings.
- All enum values must exactly match the schema options.

SCHEMA (for reference):
{schema_str}

OCR TEXT (original):
---
{raw_ocr_text}
---

Corrected JSON:"""


# ──────────────────────────────────────────────────────────────────────────────
# 2.  JSON Extraction from LLM Response
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json_from_response(text: str) -> str:
    """
    Robustly extract a JSON object from the model's raw response.

    Handles common LLM formatting failures:
      • Markdown code fences (```json ... ```)
      • Trailing prose after the closing brace
      • Leading explanation before the opening brace
      • Nested braces (finds the outermost balanced pair)
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()

    # Find the outermost balanced JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth   = 0
    in_str  = False
    escape  = False
    end_idx = start

    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

    json_str = text[start : end_idx + 1]
    return json_str


def _format_validation_errors(exc: ValidationError) -> str:
    """Format Pydantic ValidationError into a concise, LLM-readable message."""
    lines = []
    for err in exc.errors():
        loc  = " → ".join(str(l) for l in err["loc"])
        msg  = err["msg"]
        inp  = err.get("input", "<missing>")
        lines.append(f"  • Field '{loc}': {msg}  (got: {repr(inp)[:80]})")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Best-Effort Partial Payload Builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_partial_payload(
    raw_json_str:      str,
    raw_ocr_text:      str,
    warnings:          list,
    extraction_meta:   ExtractionMetadata,
) -> MedicalReportPayload:
    """
    Called after all retries are exhausted (Option 3: soft-fail).

    Strategy:
      1. Try to parse whatever JSON we have; surgically remove invalid fields.
      2. Use MedicalReportPayload.model_construct() to bypass validation
         (allows None fields, wrong types won't explode).
      3. Fall back to an empty shell if even that fails.

    The confidence in metadata will be 0.0 to signal low quality.
    """
    try:
        raw_dict: Dict[str, Any] = json.loads(raw_json_str)
    except json.JSONDecodeError:
        raw_dict = {}

    # Strip fields known to commonly cause validation failures caused by
    # hallucinated enum values — safe to null them rather than hard-crash
    _SAFE_NULL_ON_FAILURE = [
        "report_type", "patient.sex",
        "lab_panels",  "diagnoses", "medications",
    ]

    def _safe_strip(d: dict, path: str) -> None:
        keys = path.split(".")
        node = d
        for k in keys[:-1]:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return
        if isinstance(node, dict):
            node.pop(keys[-1], None)

    for path in _SAFE_NULL_ON_FAILURE:
        _safe_strip(raw_dict, path)

    # Capture secondary-doc hint before constructing payload
    extra_hint = raw_dict.pop("extraction_warnings_hint", None)
    if extra_hint:
        warnings.append(f"Secondary document detected: {extra_hint}")

    try:
        payload = MedicalReportPayload.model_construct(
            report_type  = ReportType.UNKNOWN,
            raw_ocr_text = raw_ocr_text,
            metadata     = extraction_meta,
            **{k: v for k, v in raw_dict.items()
               if k in MedicalReportPayload.model_fields},
        )
    except Exception:
        payload = MedicalReportPayload(
            report_type  = ReportType.UNKNOWN,
            raw_ocr_text = raw_ocr_text,
            metadata     = extraction_meta,
        )

    warnings.append(
        "All structuring retries exhausted. Partial payload returned. "
        "Missing fields will be imputed by the V-JEPA model."
    )
    extraction_meta.ocr_confidence = 0.0
    extraction_meta.extraction_warnings = warnings
    payload.metadata = extraction_meta
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# 4.  StructuringAgent Class
# ──────────────────────────────────────────────────────────────────────────────

class StructuringAgent:
    """
    Converts raw OCR text into a validated MedicalReportPayload.

    Uses a local Ollama LLM with prompt engineering + Pydantic validation.
    Up to 3 retry attempts with ValidationError feedback injection.
    Soft-fails gracefully if all retries are exhausted.

    Parameters
    ----------
    model : str
        Ollama model tag. Recommended: "llama3" / "llama3:8b" / "llama3.1:8b".
        For faster response on older M-series (< M2 Pro): "llama3:8b-instruct-q4_K_M"
    ollama_host : str
        Local Ollama server URL.
    max_retries : int
        Max retry attempts on ValidationError (default: 3).
    temperature : float
        LLM temperature. 0.0 = fully deterministic (best for structured extraction).
    """

    def __init__(
        self,
        model:       str = DEFAULT_MODEL,
        ollama_host: str = "http://localhost:11434",
        max_retries: int = MAX_RETRIES,
        temperature: float = 0.0,
    ) -> None:
        self.model       = model
        self.max_retries = max_retries
        self.temperature = temperature
        self._client     = ollama.Client(host=ollama_host)
        self._schema_str = _build_schema_excerpt()

        logger.info(f"[StructuringAgent] Initialized with model='{model}' max_retries={max_retries}")

    # ── Public API ───────────────────────────────────────────────────────────

    def structure(
        self,
        raw_ocr_text:    str,
        ocr_confidence:  float     = 1.0,
        preprocessing_steps: list  = None,
        deskew_angle_deg: Optional[float] = None,
        source_image_path: Optional[str]  = None,
        vision_model:    str       = "llama3.2-vision",
    ) -> MedicalReportPayload:
        """
        Parse raw OCR text into a MedicalReportPayload.

        Parameters
        ----------
        raw_ocr_text     : verbatim text from VisionAgent
        ocr_confidence   : confidence score from VisionAgent (0.0–1.0)
        preprocessing_steps : list of OpenCV step names applied
        deskew_angle_deg : rotation angle detected and corrected
        source_image_path: original image file path (for audit)
        vision_model     : VLM model name used for OCR

        Returns
        -------
        MedicalReportPayload — fully validated or best-effort partial.
        """
        t_start  = time.perf_counter()
        warnings = []

        metadata = ExtractionMetadata(
            vision_model        = vision_model,
            structuring_model   = self.model,
            ocr_confidence      = ocr_confidence,
            preprocessing_steps = preprocessing_steps or [],
            deskew_angle_deg    = deskew_angle_deg,
            source_image_path   = source_image_path,
        )

        # ── Retry loop ───────────────────────────────────────────────────────
        # Pre-focus: keep only content-rich lines to stay within llama3 context
        focused_ocr = _focus_ocr_text(raw_ocr_text)
        prompt       = _build_initial_prompt(focused_ocr, self._schema_str)
        last_json_str = ""

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"[StructuringAgent] Attempt {attempt}/{self.max_retries}")

            # 1. Call Ollama
            try:
                raw_response = self._call_ollama(prompt)
            except Exception as e:
                warnings.append(f"Ollama call failed on attempt {attempt}: {e}")
                logger.error(f"[StructuringAgent] Ollama error: {e}")
                if attempt == self.max_retries:
                    break
                time.sleep(2 ** attempt)
                continue

            # 2. Extract JSON from response
            try:
                last_json_str = _extract_json_from_response(raw_response)
            except ValueError as e:
                warnings.append(f"Attempt {attempt}: Could not find JSON in response — {e}")
                logger.warning(f"[StructuringAgent] JSON extraction failed: {e}")
                # Feed the malformed response back with instruction
                prompt = _build_retry_prompt(
                    raw_ocr_text   = raw_ocr_text,
                    schema_str     = self._schema_str,
                    previous_json  = raw_response[:2000],  # truncate for context window
                    validation_errors = f"Your response did not contain a valid JSON object. Error: {e}",
                    attempt        = attempt,
                )
                continue

            # 3. Pydantic validation
            try:
                raw_dict = json.loads(last_json_str)

                # Capture and strip the secondary-document hint before validation
                extra_hint = raw_dict.pop("extraction_warnings_hint", None)
                if extra_hint:
                    warnings.append(f"Secondary document detected: {extra_hint}")

                payload = MedicalReportPayload.model_validate(raw_dict)

                # ── SUCCESS ──────────────────────────────────────────────────
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"[StructuringAgent] ✓ Valid payload on attempt {attempt}  "
                    f"({elapsed:.1f}s)"
                )

                # Populate metadata on the successful payload
                metadata.extraction_warnings = warnings
                payload.raw_ocr_text         = raw_ocr_text
                payload.metadata             = metadata
                return payload

            except json.JSONDecodeError as e:
                val_err_str = f"Invalid JSON syntax: {e}"
                logger.warning(f"[StructuringAgent] JSONDecodeError: {e}")
                warnings.append(f"Attempt {attempt}: JSONDecodeError — {e}")

            except ValidationError as e:
                val_err_str = _format_validation_errors(e)
                logger.warning(
                    f"[StructuringAgent] ValidationError on attempt {attempt}:\n{val_err_str}"
                )
                warnings.append(
                    f"Attempt {attempt}: ValidationError — "
                    f"{len(e.errors())} field(s) invalid."
                )

            # 4. Build retry prompt with error context
            if attempt < self.max_retries:
                prompt = _build_retry_prompt(
                    raw_ocr_text       = raw_ocr_text,
                    schema_str         = self._schema_str,
                    previous_json      = last_json_str,
                    validation_errors  = val_err_str,
                    attempt            = attempt,
                )

        # ── All retries exhausted → soft-fail ────────────────────────────────
        elapsed = time.perf_counter() - t_start
        logger.warning(
            f"[StructuringAgent] All {self.max_retries} retries exhausted after "
            f"{elapsed:.1f}s. Returning partial payload."
        )
        return _build_partial_payload(
            raw_json_str    = last_json_str,
            raw_ocr_text    = raw_ocr_text,
            warnings        = warnings,
            extraction_meta = metadata,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> str:
        """Single blocking Ollama chat call. Returns the assistant message content."""
        response = self._client.chat(
            model    = self.model,
            messages = [{"role": "user", "content": prompt}],
            options  = {
                "temperature": self.temperature,
                "num_predict": 8192,   # long reports can produce large JSON
                "stop":        [],     # disable early stopping — we need the full JSON
            },
        )
        return response["message"]["content"]
