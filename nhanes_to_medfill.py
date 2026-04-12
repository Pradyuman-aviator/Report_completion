"""
nhanes_to_medfill.py
────────────────────
Downloads NHANES 2021-2023 lab files directly from CDC and saves two
CSV files into data/raw/ that slot right into the existing pipeline:

  data/raw/nhanes_anemia.csv   → loaded as an anemia-source dataset
  data/raw/nhanes_ckd.csv      → loaded as a ckd-source dataset

NHANES column → UNIFIED feature mapping
───────────────────────────────────────
  LBXHGB  → Hemoglobin     (g/dL)
  LBXMCH  → MCH            (pg)
  LBXMC   → MCHC           (g/dL)
  LBXMCVSI→ MCV            (fL)
  RIAGENDR → Gender         (1=Male, 2=Female → 0/1)
  RIDAGEYR → age
  LBXSCR  → sc  (serum creatinine)
  LBXSBU  → bu  (BUN)
  LBXSGL  → bgr (blood glucose)
  LBXSNASI→ sod (sodium)
  LBXSKSI → pot (potassium)
  LBXHCT  → pcv (hematocrit / packed cell volume)
  LBDLYMNO→ wc  (white cell count proxy via lymph absolute)
  LBXRBCSI→ rc  (red blood cell count)

  All CKD categorical fields (rbc, pc, pcc, ba, htn, dm, cad,
  appet, pe, ane) are set to NaN — they are not in NHANES.
  The imputation model will learn to fill them.

  Label logic
  ──────────
  anemia file : Hemoglobin < 12 (F) or < 13 (M)  → label=1
  ckd    file : eGFR (from creatinine) < 60       → label=1
               using CKD-EPI simplified formula

Run:
  python nhanes_to_medfill.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── output paths ─────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_ANEMIA = RAW_DIR / "nhanes_anemia.csv"
OUT_CKD    = RAW_DIR / "nhanes_ckd.csv"

# -- NHANES 2021-2023 XPT URLs ─────────────────────────────────────────────────
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
URLS = {
    "cbc"   : f"{BASE}/CBC_L.xpt",
    "biopro": f"{BASE}/BIOPRO_L.xpt",
    "demo"  : f"{BASE}/DEMO_L.xpt",
}

UNIFIED = [
    "Hemoglobin", "MCH", "MCHC", "MCV", "Gender",
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "pcv", "wc", "rc",
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]


def download_xpt(url: str, name: str) -> pd.DataFrame:
    print(f"  >> Downloading {name}  ({url})")
    try:
        df = pd.read_sas(url, format="xport", encoding="utf-8")
    except Exception as e:
        print(f"    [ERROR] Failed to read {name}: {e}")
        sys.exit(1)
    print(f"    [OK] {len(df):,} rows  {list(df.columns[:6])} ...")
    return df


def egfr(scr: pd.Series, age: pd.Series, is_female: pd.Series) -> pd.Series:
    """CKD-EPI 2021 (race-free) simplified approximation."""
    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)
    ratio = scr / kappa
    egfr = (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** (-1.200)
        * 0.9938 ** age
        * np.where(is_female, 1.012, 1.0)
    )
    return pd.Series(egfr, index=scr.index)


def main():
    print("\n===  NHANES 2021-2023  ->  MedFill format  ===\n")

    # ── 1. Download ───────────────────────────────────────────────────────────
    cbc    = download_xpt(URLS["cbc"],    "CBC (blood count)")
    biopro = download_xpt(URLS["biopro"], "BIOPRO (metabolic)")
    demo   = download_xpt(URLS["demo"],   "DEMO")

    # ── 2. Merge on SEQN (participant ID) ────────────────────────────────────
    df = (
        demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
        .merge(cbc[[  "SEQN",
                       "LBXHGB",    # Hemoglobin
                       "LBXMCHSI",  # MCH
                       "LBXMC",     # MCHC
                       "LBXMCVSI",  # MCV
                       "LBXHCT",    # Hematocrit / pcv
                       "LBXRBCSI",  # RBC count
                       "LBXWBCSI",  # WBC count (wc)
                   ]], on="SEQN", how="inner")
        .merge(biopro[["SEQN",
                        "LBXSCR",   # Serum creatinine
                        "LBXSBU",   # BUN
                        "LBXSGL",   # Blood glucose
                        "LBXSNASI", # Sodium
                        "LBXSKSI",  # Potassium
                       ]], on="SEQN", how="inner")
    )
    print(f"\n  Merged shape: {df.shape}")

    # ── 3. Rename to UNIFIED names ─────────────────────────────────────────────
    df = df.rename(columns={
        "LBXHGB"    : "Hemoglobin",
        "LBXMCHSI"  : "MCH",
        "LBXMC"     : "MCHC",
        "LBXMCVSI"  : "MCV",
        "RIAGENDR"  : "Gender",     # 1=Male, 2=Female -> recode below
        "RIDAGEYR"  : "age",
        "LBXSCR"    : "sc",
        "LBXSBU"    : "bu",
        "LBXSGL"    : "bgr",
        "LBXSNASI"  : "sod",
        "LBXSKSI"   : "pot",
        "LBXHCT"    : "pcv",
        "LBXWBCSI"  : "wc",
        "LBXRBCSI"  : "rc",
    })

    # Gender: NHANES 1=Male 2=Female → our convention: 1=Male 0=Female
    df["Gender"] = (df["Gender"] == 1).astype(float)

    # bp not in basic NHANES lab files — leave as NaN
    df["bp"] = np.nan

    # All CKD-only categorical features → NaN (model will impute)
    for col in ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]:
        df[col] = np.nan

    # ── 4. Build ANEMIA file ───────────────────────────────────────────────────
    # WHO anemia threshold: Hgb < 12 g/dL (female) or < 13 g/dL (male)
    hgb      = df["Hemoglobin"]
    is_male  = df["Gender"] == 1
    anemia_label = np.where(
        is_male,
        (hgb < 13).astype(int),
        (hgb < 12).astype(int),
    )
    anemia_df = df[UNIFIED].copy()
    anemia_df["label"]   = anemia_label
    anemia_df["dataset"] = "nhanes_anemia"
    anemia_df = anemia_df.dropna(subset=["Hemoglobin"])

    anemia_df.to_csv(OUT_ANEMIA, index=False)
    n_pos = anemia_df["label"].sum()
    print(f"  [OK] Anemia CSV  -> {OUT_ANEMIA}")
    print(f"    Rows={len(anemia_df):,}  Anemic={int(n_pos):,}  "
          f"({100*n_pos/len(anemia_df):.1f}%)")

    # ── 5. Build CKD file ─────────────────────────────────────────────────────
    # eGFR < 60 mL/min/1.73m² = CKD (stages 3-5)
    is_female = df["Gender"] == 0
    df_ckd    = df.dropna(subset=["sc", "age"]).copy()
    is_female_ckd = is_female.loc[df_ckd.index]
    df_ckd["_egfr"] = egfr(df_ckd["sc"], df_ckd["age"], is_female_ckd)

    ckd_df = df_ckd[UNIFIED].copy()
    ckd_df["label"]   = (df_ckd["_egfr"] < 60).astype(int)
    ckd_df["dataset"] = "nhanes_ckd"

    ckd_df.to_csv(OUT_CKD, index=False)
    n_ckd = ckd_df["label"].sum()
    print(f"  [OK] CKD CSV     -> {OUT_CKD}")
    print(f"    Rows={len(ckd_df):,}  CKD={int(n_ckd):,}  "
          f"({100*n_ckd/len(ckd_df):.1f}%)")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    total = len(anemia_df) + len(ckd_df)
    print(f"  Total NHANES rows added : {total:,}")
    print(f"  Your original data      : 1,821  (anemia + CKD CSVs)")
    print(f"  Combined potential      : ~{total + 1821:,} rows")
    print("\n  Next step -> run audit.py to see the new dataset size.")
    print("===  Done  ===\n")


if __name__ == "__main__":
    main()
