"""
infer.py
=========
MedFill End-to-End Inference
-----------------------------

Give it a medical report image → get back a complete lab panel with ALL
missing values predicted by the trained imputation model.

Usage:
    python infer.py path/to/report.jpg
    python infer.py path/to/report.jpg --checkpoint ml_pipeline/checkpoints/best_model.pt
    python infer.py path/to/report.jpg --no-ollama   # skip OCR, use manual input

Flow:
    report.jpg
        │
        ▼  VisionAgent (llava:7b OCR)
        │
        ▼  StructuringAgent (llama3 → structured JSON)
        │
        ▼  map lab values → 25-column unified schema
        │
        ▼  TabularImputerModel (trained on Anemia + CKD data)
        │
        ▼  RESULT: complete lab panel with predicted missing values
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

# ── Feature schema (must match dataset.py) ───────────────────────────────────
UNIFIED = [
    "Hemoglobin", "MCH", "MCHC", "MCV", "Gender",
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "pcv", "wc", "rc",
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]
NUM_FEATURES = len(UNIFIED)

# Mapping: lab test name keywords  →  UNIFIED column name
# (case-insensitive partial match)
LAB_NAME_MAP = {
    "haemoglobin":  "Hemoglobin",
    "hemoglobin":   "Hemoglobin",
    "hb":           "Hemoglobin",
    "mch":          "MCH",
    "mchc":         "MCHC",
    "mcv":          "MCV",
    "gender":       "Gender",
    "sex":          "Gender",
    "age":          "age",
    "blood pressure": "bp",
    "bp":           "bp",
    "blood glucose": "bgr",
    "glucose":      "bgr",
    "bgr":          "bgr",
    "blood urea":   "bu",
    "urea":         "bu",
    "bu":           "bu",
    "serum creatinine": "sc",
    "creatinine":   "sc",
    "sc":           "sc",
    "sodium":       "sod",
    "sod":          "sod",
    "potassium":    "pot",
    "pot":          "pot",
    "packed cell":  "pcv",
    "pcv":          "pcv",
    "white blood":  "wc",
    "wbc":          "wc",
    "leucocytes":   "wc",
    "wc":           "wc",
    "red blood cell count": "rc",
    "rbc count":    "rc",
    "erythrocytes": "rc",
    "rc":           "rc",
    "red blood cells": "rbc",   # categorical
    "rbc":          "rbc",
    "pus cells":    "pc",
    "pc":           "pc",
    "pus cell clumps": "pcc",
    "pcc":          "pcc",
    "bacteria":     "ba",
    "ba":           "ba",
    "hypertension": "htn",
    "htn":          "htn",
    "diabetes":     "dm",
    "dm":           "dm",
    "coronary":     "cad",
    "cad":          "cad",
    "appetite":     "appet",
    "appet":        "appet",
    "pedal edema":  "pe",
    "edema":        "pe",
    "pe":           "pe",
    "anaemia":      "ane",
    "anemia":       "ane",
    "ane":          "ane",
}

# Categorical → numeric mapping (same as dataset.py BMAP)
BMAP = {
    "yes": 1, "no": 0,
    "normal": 1, "abnormal": 0,
    "present": 1, "notpresent": 0,
    "good": 1, "poor": 0,
    "male": 1, "female": 0,
}

# Binary / categorical columns — display as labels not floats
BINARY_COLS = {"rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "pe", "ane"}
APPET_COL   = "appet"   # good / poor
GENDER_COL  = "Gender"  # male / female

# Approximate normal ranges for ⚠ warnings (min, max)
NORMAL_RANGES = {
    "Hemoglobin": (10.0, 18.0),
    "MCH":        (25.0, 35.0),
    "MCHC":       (30.0, 38.0),
    "MCV":        (60.0, 100.0),
    "age":        (1.0,  120.0),
    "bp":         (50.0, 200.0),
    "bgr":        (40.0, 600.0),
    "bu":         (5.0,  200.0),
    "sc":         (0.4,  15.0),
    "sod":        (120.0, 160.0),
    "pot":        (2.5,   7.0),
    "pcv":        (15.0,  60.0),
    "wc":         (2000.0, 20000.0),
    "rc":         (2.0,   8.0),
}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model (copied from train.py — keep in sync)
# ──────────────────────────────────────────────────────────────────────────────

class TabularImputerModel(torch.nn.Module):
    def __init__(self, num_features=NUM_FEATURES, embed_dim=64, depth=3,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_embed = torch.nn.Linear(1, embed_dim)
        self.pos_embed     = torch.nn.Embedding(num_features, embed_dim)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.output_head = torch.nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, F = x.shape
        tokens = self.feature_embed(x.unsqueeze(-1))
        positions = torch.arange(F, device=x.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(positions)
        tokens = self.transformer(tokens)
        return self.output_head(tokens).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Normalization stats from training data
# ──────────────────────────────────────────────────────────────────────────────

def load_normalization_stats(raw_dir: str = "data/raw"):
    """Compute mean/std from the same CSVs used in training."""
    from ml_pipeline.data.dataset import _load_anemia, _load_ckd, UNIFIED
    raw = Path(raw_dir)
    a = _load_anemia(raw / "anemia.csv")
    b = _load_ckd(raw / "kidney_disease.csv")
    merged = pd.concat([a, b], ignore_index=True)
    for c in UNIFIED:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float32")
    means = merged[UNIFIED].mean()
    stds  = merged[UNIFIED].std().replace(0, 1)
    return means, stds


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Map MedicalReportPayload → UNIFIED feature vector
# ──────────────────────────────────────────────────────────────────────────────

def payload_to_features(payload_dict: dict) -> Dict[str, Optional[float]]:
    """
    Extract known values from the structured JSON payload into a
    {column_name: value_or_None} dict aligned to UNIFIED.
    """
    features: Dict[str, Optional[float]] = {col: None for col in UNIFIED}

    # ── Patient demographics ──────────────────────────────────────────────
    patient = payload_dict.get("patient") or {}
    if patient.get("age_years") is not None:
        features["age"] = float(patient["age_years"])
    sex = (patient.get("sex") or "").lower().strip()
    if sex in ("male", "m", "1"):
        features["Gender"] = 1.0
    elif sex in ("female", "f", "0"):
        features["Gender"] = 0.0
    elif sex in BMAP:
        features["Gender"] = float(BMAP[sex])

    # ── Vitals ────────────────────────────────────────────────────────────
    vitals = payload_dict.get("vitals") or {}
    if vitals.get("blood_pressure_systolic") is not None:
        features["bp"] = float(vitals["blood_pressure_systolic"])

    # ── Lab panels ────────────────────────────────────────────────────────
    for panel in payload_dict.get("lab_panels", []):
        for result in panel.get("results", []):
            name  = (result.get("test_name") or "").lower().strip()
            value = result.get("value")

            # Match test name to UNIFIED column
            matched_col = None
            for keyword, col in LAB_NAME_MAP.items():
                if keyword in name:
                    matched_col = col
                    break

            if matched_col is None:
                continue

            if value is not None:
                features[matched_col] = float(value)
            else:
                # Handle categorical text values
                val_text = (result.get("value_text") or "").lower().strip()
                if val_text in BMAP:
                    features[matched_col] = float(BMAP[val_text])

    return features


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Run imputation
# ──────────────────────────────────────────────────────────────────────────────

def impute(
    known_values: Dict[str, Optional[float]],
    checkpoint:   str   = "ml_pipeline/checkpoints/best_model.pt",
    raw_dir:      str   = "data/raw",
    device_str:   str   = "auto",
) -> Dict[str, dict]:
    """
    Run the trained TabularImputerModel on a dict of known lab values.

    Returns a dict: column → {"value": float, "predicted": bool}
    """
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load model
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TabularImputerModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load normalisation stats
    means, stds = load_normalization_stats(raw_dir)

    # Build normalized input vector
    raw_vec  = np.array([known_values.get(c) for c in UNIFIED], dtype="float32")
    obs_mask = (~np.isnan(raw_vec)).astype("float32")

    # Normalize observed values
    norm_vec = raw_vec.copy()
    for i, col in enumerate(UNIFIED):
        if obs_mask[i] == 1:
            norm_vec[i] = (raw_vec[i] - means[col]) / stds[col]
        else:
            norm_vec[i] = 0.0   # masked → zero

    x = torch.from_numpy(norm_vec).unsqueeze(0).to(device)  # (1, F)

    with torch.no_grad():
        pred_norm = model(x).squeeze(0).cpu().numpy()        # (F,)

    # Denormalize all predictions
    pred_raw = pred_norm * stds.values + means.values

    # Build result: keep observed values; fill in predicted for missing
    result = {}
    for i, col in enumerate(UNIFIED):
        if obs_mask[i] == 1:
            result[col] = {"value": round(float(raw_vec[i]), 4), "predicted": False}
        else:
            result[col] = {"value": round(float(pred_raw[i]), 4), "predicted": True}

    return result


def format_value(col: str, val: float, predicted: bool) -> str:
    """Format a column value as a human-readable string."""
    if col in BINARY_COLS:
        label = "yes" if val >= 0.5 else "no"
        if col in ("rbc", "pc"):
            label = "normal" if val >= 0.5 else "abnormal"
        elif col in ("pcc", "ba"):
            label = "present" if val >= 0.5 else "notpresent"
        return label
    elif col == APPET_COL:
        return "good" if val >= 0.5 else "poor"
    elif col == GENDER_COL:
        return "male" if val >= 0.5 else "female"
    else:
        return f"{val:.3f}"


def print_panel(complete: dict) -> None:
    """Pretty-print the complete lab panel with warnings for out-of-range values."""
    print(f"  {'Feature':<20} {'Value':>12}  {'Source':>12}  Note")
    print(f"  {'-'*20} {'-'*12}  {'-'*12}  ----")
    for col, info in complete.items():
        val  = info["value"]
        src  = "★ PREDICTED" if info["predicted"] else "  extracted"
        disp = format_value(col, val, info["predicted"])

        # Range warning (only for numeric observed values)
        note = ""
        if not info["predicted"] and col in NORMAL_RANGES:
            lo, hi = NORMAL_RANGES[col]
            if val < lo or val > hi:
                note = f"⚠ out of range ({lo}–{hi})"

        print(f"  {col:<20} {disp:>12}  {src:>12}  {note}")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full pipeline: image → complete result
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    image_path:  str,
    checkpoint:  str = "ml_pipeline/checkpoints/best_model.pt",
    raw_dir:     str = "data/raw",
) -> dict:
    """
    Full end-to-end:
        image → OCR → structured JSON → imputation → complete result
    """
    print(f"\n[MedFill] Processing: {image_path}")
    print("=" * 60)

    # ── Stage 1: AI agents (OCR + structuring) ────────────────────────────
    print("[1/3] Running AI agents (OCR + structuring)...")
    from ai_agents.pipeline import MedicalReportPipeline
    pipe    = MedicalReportPipeline()
    payload = pipe.process(image_path)
    payload_dict = payload.model_dump(mode="json")

    print(f"  Report type : {payload.report_type.value}")
    print(f"  Patient     : {payload.patient.name if payload.patient else 'Unknown'}")
    n_results = sum(len(p.results) for p in payload.lab_panels)
    print(f"  Lab results found: {n_results}")

    # ── Stage 2: Extract known features ──────────────────────────────────
    print("[2/3] Mapping to tabular features...")
    known = payload_to_features(payload_dict)
    n_known   = sum(1 for v in known.values() if v is not None)
    n_missing = NUM_FEATURES - n_known
    print(f"  Known features  : {n_known}/{NUM_FEATURES}")
    print(f"  Missing features: {n_missing}/{NUM_FEATURES}  ← will be predicted")

    # ── Stage 3: Imputation ───────────────────────────────────────────────
    print("[3/3] Running imputation model...")
    complete = impute(known, checkpoint=checkpoint, raw_dir=raw_dir)

    # ── Print result ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  COMPLETE LAB PANEL")
    print("=" * 72)
    print_panel(complete)
    print("=" * 72)

    n_predicted = sum(1 for v in complete.values() if v["predicted"])
    print(f"\n  Extracted : {NUM_FEATURES - n_predicted} values")
    print(f"  Predicted : {n_predicted} values  (★)")

    return {
        "payload":          payload_dict,
        "complete_panel":   complete,
        "stats": {
            "n_extracted":  NUM_FEATURES - n_predicted,
            "n_predicted":  n_predicted,
            "total_features": NUM_FEATURES,
        }
    }


def run_manual_input(
    checkpoint: str = "ml_pipeline/checkpoints/best_model.pt",
    raw_dir:    str = "data/raw",
) -> dict:
    """
    Interactive mode: type known lab values manually, imputer fills the rest.
    Useful for testing without Ollama.
    """
    print("\n[MedFill] Manual Input Mode")
    print("=" * 60)
    # Hint text for each field
    HINTS = {
        "Gender": "M/F or 1/0",
        "rbc":    "normal/abnormal or 1/0",
        "pc":     "normal/abnormal or 1/0",
        "pcc":    "present/notpresent or 1/0",
        "ba":     "present/notpresent or 1/0",
        "htn":    "yes/no or 1/0",
        "dm":     "yes/no or 1/0",
        "cad":    "yes/no or 1/0",
        "appet":  "good/poor or 1/0",
        "pe":     "yes/no or 1/0",
        "ane":    "yes/no or 1/0",
    }
    print("Enter known lab values (press Enter to skip / leave blank):\n")

    # Gender alias map
    GENDER_MAP = {"m": 1.0, "male": 1.0, "f": 0.0, "female": 0.0}

    known: Dict[str, Optional[float]] = {col: None for col in UNIFIED}
    for col in UNIFIED:
        hint = f"  ({HINTS[col]})" if col in HINTS else ""
        raw = input(f"  {col}{hint}: ").strip()
        if raw:
            try:
                rl = raw.lower()
                if col == "Gender" and rl in GENDER_MAP:
                    known[col] = GENDER_MAP[rl]
                elif rl in BMAP:
                    known[col] = float(BMAP[rl])
                else:
                    known[col] = float(raw)
            except ValueError:
                print(f"    ⚠ Could not parse '{raw}' — leaving as missing")

    n_known = sum(1 for v in known.values() if v is not None)
    print(f"\n  Known: {n_known}/{NUM_FEATURES}  |  Predicting: {NUM_FEATURES - n_known}")

    complete = impute(known, checkpoint=checkpoint, raw_dir=raw_dir)

    print("\n" + "=" * 72)
    print("  COMPLETE LAB PANEL")
    print("=" * 72)
    print_panel(complete)
    print("=" * 72)

    return complete


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedFill: report image → complete lab panel with predicted values"
    )
    parser.add_argument("image", nargs="?", help="Path to medical report image (JPEG/PNG)")
    parser.add_argument("--checkpoint", default="ml_pipeline/checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--raw-dir",    default="data/raw",
                        help="Path to raw CSV directory (for normalization stats)")
    parser.add_argument("--out",        default=None,
                        help="Save full result JSON to this file")
    parser.add_argument("--manual",     action="store_true",
                        help="Enter known values manually instead of using OCR")
    args = parser.parse_args()

    if args.manual:
        result = run_manual_input(
            checkpoint = args.checkpoint,
            raw_dir    = args.raw_dir,
        )
    elif args.image:
        result = run_full_pipeline(
            image_path = args.image,
            checkpoint = args.checkpoint,
            raw_dir    = args.raw_dir,
        )
    else:
        parser.print_help()
        sys.exit(1)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[MedFill] Saved → {args.out}")
