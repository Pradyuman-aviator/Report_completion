"""
ai_agents/schemas.py
=====================
Pydantic Models for Structured Medical Report Data
----------------------------------------------------

This file is the single source of truth for the JSON schema that flows
between every layer of the system:

    Android Camera
        ↓
    vision_agent.py     → raw OCR text
        ↓
    structuring_agent.py → validates against these Pydantic models
        ↓
    api_gateway          → receives a MedicalReportPayload JSON body
        ↓
    ml_pipeline          → consumes structured fields for inference

Design principles:
  • Every field has a clear Optional/Required declaration.
  • Enums enforce controlled vocabulary (prevents hallucinated values from LLMs).
  • All numeric lab values are typed as float with explicit units captured
    in a companion string field — never store "10.2 mg/dL" as a single string.
  • Dates are always ISO-8601 strings (YYYY-MM-DD) — LLMs handle these reliably.
  • The top-level MedicalReportPayload has a `confidence` field so downstream
    components can threshold on extraction quality.

Usage:
    >>> from ai_agents.schemas import MedicalReportPayload
    >>> payload = MedicalReportPayload(**extracted_dict)
    >>> print(payload.model_dump_json(indent=2))
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Controlled Vocabulary Enums
# ──────────────────────────────────────────────────────────────────────────────

class ReportType(str, Enum):
    """The category of the scanned medical document."""
    BLOOD_TEST       = "blood_test"
    URINE_TEST       = "urine_test"
    IMAGING_REPORT   = "imaging_report"    # X-ray, MRI, CT, ultrasound
    PATHOLOGY        = "pathology"
    PRESCRIPTION     = "prescription"
    DISCHARGE_SUMMARY= "discharge_summary"
    CONSULTATION_NOTE= "consultation_note"
    ECG_REPORT       = "ecg_report"
    VACCINATION_RECORD= "vaccination_record"
    UNKNOWN          = "unknown"


class Sex(str, Enum):
    MALE    = "male"
    FEMALE  = "female"
    OTHER   = "other"
    UNKNOWN = "unknown"


class AbnormalityFlag(str, Enum):
    """Whether a lab value is within the reference range."""
    NORMAL   = "normal"
    HIGH     = "high"
    LOW      = "low"
    CRITICAL = "critical"
    UNKNOWN  = "unknown"


class Severity(str, Enum):
    MILD     = "mild"
    MODERATE = "moderate"
    SEVERE   = "severe"
    UNKNOWN  = "unknown"


class MedicationRoute(str, Enum):
    ORAL       = "oral"
    INTRAVENOUS= "intravenous"
    TOPICAL    = "topical"
    INHALED    = "inhaled"
    SUBCUTANEOUS = "subcutaneous"
    OTHER      = "other"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Nested Sub-Models
# ──────────────────────────────────────────────────────────────────────────────

class PatientInfo(BaseModel):
    """Demographics extracted from the report header."""
    name:           Optional[str]   = Field(None,  description="Full patient name as printed on report.")
    patient_id:     Optional[str]   = Field(None,  description="Hospital / clinic patient ID or MRN.")
    date_of_birth:  Optional[str]   = Field(None,  description="ISO-8601 date: YYYY-MM-DD.")
    age_years:      Optional[float] = Field(None,  ge=0, le=150)
    sex:            Optional[Sex]   = Field(Sex.UNKNOWN)
    contact_number: Optional[str]   = Field(None)
    address:        Optional[str]   = Field(None)

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def normalise_dob(cls, v: Optional[str]) -> Optional[str]:
        """Accept common Indian date formats and convert to ISO-8601."""
        if v is None:
            return None
        import re
        v = v.strip()
        # DD/MM/YYYY  or  DD-MM-YYYY  →  YYYY-MM-DD
        m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", v)
        if m:
            d, mo, yr = m.groups()
            return f"{yr}-{mo.zfill(2)}-{d.zfill(2)}"
        return v


class FacilityInfo(BaseModel):
    """The hospital / clinic that issued the report."""
    name:          Optional[str] = Field(None, description="Facility / lab name.")
    address:       Optional[str] = Field(None)
    phone:         Optional[str] = Field(None)
    accreditation: Optional[str] = Field(None, description="e.g. NABL, NABH, CAP.")


class DoctorInfo(BaseModel):
    """The referring or reporting clinician."""
    name:           Optional[str] = Field(None, description="Dr. <Name> as on report.")
    specialisation: Optional[str] = Field(None)
    registration_no: Optional[str] = Field(None, description="Medical council registration number.")
    signature_present: bool = Field(False)


class LabResult(BaseModel):
    """
    A single measured laboratory parameter.

    Examples:
        Haemoglobin: 11.2 g/dL  (LOW — ref 13.0–17.0)
        Fasting Glucose: 5.8 mmol/L (NORMAL — ref 3.9–6.1)
    """
    test_name:       str            = Field(...,  description="Parameter name exactly as printed.")
    value:           Optional[float]= Field(None, description="Numeric value. None if non-numeric (e.g. 'Positive').")
    value_text:      Optional[str]  = Field(None, description="Raw string value (fallback for non-numeric results).")
    unit:            Optional[str]  = Field(None, description="Measurement unit, e.g. 'g/dL', 'mmol/L', '%'.")
    reference_low:   Optional[float]= Field(None, description="Lower bound of normal reference range.")
    reference_high:  Optional[float]= Field(None, description="Upper bound of normal reference range.")
    reference_text:  Optional[str]  = Field(None, description="Full reference range string, e.g. '13.0 – 17.0'.")
    flag:            AbnormalityFlag= Field(AbnormalityFlag.UNKNOWN)
    method:          Optional[str]  = Field(None, description="Measurement method if stated, e.g. 'Spectrophotometry'.")

    @model_validator(mode="after")
    def auto_flag(self) -> "LabResult":
        """Auto-compute flag from value + reference range if not set by LLM."""
        if self.flag != AbnormalityFlag.UNKNOWN:
            return self
        if self.value is not None:
            if self.reference_high is not None and self.value > self.reference_high:
                self.flag = AbnormalityFlag.HIGH
            elif self.reference_low is not None and self.value < self.reference_low:
                self.flag = AbnormalityFlag.LOW
            elif self.reference_low is not None and self.reference_high is not None:
                self.flag = AbnormalityFlag.NORMAL
        return self


class LabPanel(BaseModel):
    """
    A named group of LabResults (e.g. 'Complete Blood Count', 'Lipid Profile').
    Most Indian diagnostic reports group tests into panels.
    """
    panel_name: str              = Field(..., description="e.g. 'CBC', 'LFT', 'KFT', 'Thyroid Profile'.")
    results:    List[LabResult]  = Field(default_factory=list)


class Diagnosis(BaseModel):
    """A clinical diagnosis or impression stated in the report."""
    name:        str                = Field(..., description="Diagnosis name or ICD-10 description.")
    icd10_code:  Optional[str]      = Field(None, description="ICD-10 code if extractable.")
    severity:    Severity           = Field(Severity.UNKNOWN)
    is_primary:  bool               = Field(False, description="True if marked as the principal diagnosis.")
    notes:       Optional[str]      = Field(None)


class Medication(BaseModel):
    """A single prescribed drug from a prescription or discharge summary."""
    drug_name:   str                     = Field(..., description="Generic or brand name as written.")
    dosage:      Optional[str]           = Field(None, description="e.g. '500 mg', '10 units'.")
    frequency:   Optional[str]           = Field(None, description="e.g. 'twice daily', 'OD', 'TID'.")
    duration:    Optional[str]           = Field(None, description="e.g. '5 days', '1 month'.")
    route:       MedicationRoute         = Field(MedicationRoute.ORAL)
    instructions:Optional[str]           = Field(None, description="e.g. 'Take after meals'.")


class VitalSigns(BaseModel):
    """Physical examination findings recorded on the report."""
    blood_pressure_systolic:  Optional[float] = Field(None, description="mmHg")
    blood_pressure_diastolic: Optional[float] = Field(None, description="mmHg")
    heart_rate_bpm:           Optional[float] = Field(None)
    temperature_celsius:      Optional[float] = Field(None)
    spo2_percent:             Optional[float] = Field(None, ge=0, le=100)
    weight_kg:                Optional[float] = Field(None)
    height_cm:                Optional[float] = Field(None)
    bmi:                      Optional[float] = Field(None)

    @model_validator(mode="after")
    def compute_bmi(self) -> "VitalSigns":
        if self.bmi is None and self.weight_kg and self.height_cm:
            h_m = self.height_cm / 100.0
            self.bmi = round(self.weight_kg / (h_m ** 2), 1)
        return self


class ImagingFinding(BaseModel):
    """Observations from a radiology or ultrasound report."""
    modality:     Optional[str] = Field(None, description="e.g. 'X-Ray', 'USG', 'MRI', 'CT'.")
    region:       Optional[str] = Field(None, description="e.g. 'Chest PA View', 'Abdomen'.")
    findings:     Optional[str] = Field(None, description="Full findings text as extracted.")
    impression:   Optional[str] = Field(None, description="Radiologist's final impression / conclusion.")


class ExtractionMetadata(BaseModel):
    """
    Provenance and quality information for audit and downstream thresholding.
    Populated by vision_agent.py and structuring_agent.py.
    """
    vision_model:        str   = Field("llama3.2-vision", description="Ollama VLM used for OCR.")
    structuring_model:   str   = Field("llama3",          description="Ollama LLM used for JSON structuring.")
    ocr_confidence:      float = Field(0.0, ge=0.0, le=1.0,
                                       description="Self-reported confidence from VLM (0–1).")
    preprocessing_steps: List[str] = Field(
        default_factory=list,
        description="OpenCV operations applied: e.g. ['grayscale','denoise','threshold','deskew']."
    )
    deskew_angle_deg:    Optional[float] = Field(None, description="Detected skew angle in degrees.")
    source_image_path:   Optional[str]   = Field(None)
    report_language:     str             = Field("en",  description="ISO 639-1 language code.")
    extraction_warnings: List[str]       = Field(
        default_factory=list,
        description="Non-fatal issues observed during extraction (e.g. 'Seal obscures bottom-right corner')."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Top-Level Payload
# ──────────────────────────────────────────────────────────────────────────────

class MedicalReportPayload(BaseModel):
    """
    The canonical data contract for the entire pipeline.

    This object is:
      • Produced by  structuring_agent.py
      • Consumed by  api_gateway/main.py  (validated again as a request body)
      • Fed into     ml_pipeline/data/dataset.py  for inference / training

    All fields except `report_type` and `metadata` are Optional to handle
    the diversity of real-world Indian medical report formats gracefully.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    report_type:    ReportType          = Field(ReportType.UNKNOWN)
    report_date:    Optional[str]       = Field(None, description="ISO-8601: YYYY-MM-DD.")
    report_id:      Optional[str]       = Field(None, description="Lab / hospital report reference number.")
    accession_no:   Optional[str]       = Field(None, description="Radiology accession number if present.")

    # ── Entities ──────────────────────────────────────────────────────────
    patient:        Optional[PatientInfo]   = None
    facility:       Optional[FacilityInfo]  = None
    referring_doctor: Optional[DoctorInfo]  = None
    reporting_doctor: Optional[DoctorInfo]  = None

    # ── Clinical Content ──────────────────────────────────────────────────
    vitals:         Optional[VitalSigns]        = None
    lab_panels:     List[LabPanel]              = Field(default_factory=list)
    diagnoses:      List[Diagnosis]             = Field(default_factory=list)
    medications:    List[Medication]            = Field(default_factory=list)
    imaging:        Optional[ImagingFinding]    = None
    clinical_summary: Optional[str]            = Field(
        None, description="Free-text clinical notes / chief complaint / history."
    )
    follow_up_instructions: Optional[str]      = Field(None)

    # ── Raw OCR (kept for audit / reprocessing) ───────────────────────────
    raw_ocr_text:   Optional[str]   = Field(
        None, description="Verbatim text as returned by the vision agent before structuring."
    )

    # ── Provenance ────────────────────────────────────────────────────────
    metadata:       ExtractionMetadata = Field(default_factory=ExtractionMetadata)

    model_config = {
        "json_schema_extra": {
            "example": {
                "report_type": "blood_test",
                "report_date": "2024-03-15",
                "patient": {"name": "Ravi Kumar", "age_years": 45, "sex": "male"},
                "lab_panels": [{
                    "panel_name": "CBC",
                    "results": [{
                        "test_name": "Haemoglobin",
                        "value": 11.2,
                        "unit": "g/dL",
                        "reference_low": 13.0,
                        "reference_high": 17.0,
                        "flag": "low"
                    }]
                }]
            }
        }
    }
