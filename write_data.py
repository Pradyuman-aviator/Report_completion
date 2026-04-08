"""Helper script — run once to generate ml_pipeline/data/ files."""
from pathlib import Path

root = Path(__file__).parent

# Create directory
(root / "ml_pipeline" / "data").mkdir(parents=True, exist_ok=True)

# ── __init__.py ──────────────────────────────────────────────────────────────
(root / "ml_pipeline" / "data" / "__init__.py").write_text(
    "from ml_pipeline.data.dataset import build_loaders, MedicalTabularDataset\n"
    "__all__ = ['build_loaders', 'MedicalTabularDataset']\n"
)

# ── dataset.py ───────────────────────────────────────────────────────────────
dataset_code = r"""
import warnings, numpy as np, pandas as pd, torch
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings("ignore")

UNIFIED = [
    "Hemoglobin","MCH","MCHC","MCV","Gender",
    "age","bp","bgr","bu","sc","sod","pot","pcv","wc","rc",
    "rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane",
]
NUM_FEATURES = len(UNIFIED)

CKD_NUM = ["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
CKD_CAT = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
BMAP    = {"yes":1,"no":0,"normal":1,"abnormal":0,
           "present":1,"notpresent":0,"good":1,"poor":0,"ckd":1,"notckd":0}


def _load_anemia(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df["label"]   = df["Result"].astype(int)
    df["dataset"] = "anemia"
    for c in UNIFIED:
        if c not in df.columns:
            df[c] = np.nan
    return df[UNIFIED + ["label", "dataset"]]


def _load_ckd(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    for col in CKD_CAT + ["classification"]:
        if col in df.columns:
            df[col] = df[col].str.lower().map(BMAP)
    for col in CKD_NUM:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"]   = pd.to_numeric(df["classification"], errors="coerce")
    df["dataset"] = "ckd"
    for c in UNIFIED:
        if c not in df.columns:
            df[c] = np.nan
    return df[UNIFIED + ["label", "dataset"]]


def _normalise(merged: pd.DataFrame) -> pd.DataFrame:
    means = merged[UNIFIED].mean()
    stds  = merged[UNIFIED].std().replace(0, 1)
    df    = merged.copy()
    for c in UNIFIED:
        ok = df[c].notna()
        df.loc[ok, c] = (df.loc[ok, c] - means[c]) / stds[c]
    return df


class MedicalTabularDataset(Dataset):
    # Tabular dataset with random mask augmentation for VQ-VAE imputation.

    def __init__(self, df: pd.DataFrame, mask_ratio: float = 0.30, augment: bool = True):
        self.mask_ratio = mask_ratio
        self.augment    = augment
        self.features   = df[UNIFIED].values.astype("float32")
        self.labels     = df["label"].values.astype("int64")
        self.datasets   = df["dataset"].values
        self.obs_mask   = (~np.isnan(self.features)).astype("float32")
        self.filled     = np.where(np.isnan(self.features), 0.0, self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        x   = self.filled[idx].copy()
        obs = self.obs_mask[idx].copy()
        if self.augment and self.mask_ratio > 0:
            present = np.where(obs == 1)[0]
            n = max(1, int(len(present) * self.mask_ratio))
            obs[np.random.choice(present, n, replace=False)] = 0
        return {
            "masked_x":   torch.from_numpy(x * obs),
            "original_x": torch.from_numpy(x),
            "mask":        torch.from_numpy(obs),
            "obs_mask":    torch.from_numpy(self.obs_mask[idx].copy()),
            "label":       torch.tensor(self.labels[idx], dtype=torch.long),
            "dataset":     self.datasets[idx],
        }


def build_loaders(
    raw_dir:     str   = "data/raw",
    val_split:   float = 0.15,
    mask_ratio:  float = 0.30,
    batch_size:  int   = 32,
    num_workers: int   = 0,
    seed:        int   = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    raw = Path(raw_dir)
    a   = _load_anemia(raw / "anemia.csv")
    b   = _load_ckd(raw / "kidney_disease.csv")
    merged = pd.concat([a, b], ignore_index=True)
    for c in UNIFIED:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float32")
    merged = merged.dropna(subset=["label"])
    merged = _normalise(merged)
    tr, va = train_test_split(
        merged, test_size=val_split, random_state=seed, stratify=merged["label"]
    )
    tds = MedicalTabularDataset(tr, mask_ratio=mask_ratio, augment=True)
    vds = MedicalTabularDataset(va, mask_ratio=0.0,        augment=False)
    kw  = dict(num_workers=num_workers, pin_memory=True)
    meta = {
        "num_features":  NUM_FEATURES,
        "feature_names": UNIFIED,
        "train_size":    len(tds),
        "val_size":      len(vds),
    }
    print(f"[dataset] Train={len(tds)} | Val={len(vds)} | Features={NUM_FEATURES}")
    return (
        DataLoader(tds, batch_size, shuffle=True,  **kw),
        DataLoader(vds, batch_size, shuffle=False, **kw),
        meta,
    )


if __name__ == "__main__":
    tl, vl, m = build_loaders()
    b = next(iter(tl))
    print("masked_x:", b["masked_x"].shape)
    print("label:   ", b["label"][:4])
    print("[PASS]")
""".lstrip()

(root / "ml_pipeline" / "data" / "dataset.py").write_text(dataset_code)

print("Done! Files written:")
print("  ml_pipeline/data/__init__.py")
print("  ml_pipeline/data/dataset.py")
