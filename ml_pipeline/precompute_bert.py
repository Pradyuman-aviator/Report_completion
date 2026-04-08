"""
ml_pipeline/precompute_bert.py
================================
ClinicalBERT Embedding Pre-computation Script
---------------------------------------------

Reads the CSV manifest, passes each row's clinical note text through
`emilyalsentzer/Bio_ClinicalBERT` (frozen), and saves the [CLS] token
embedding as a .pt file alongside the scanned images.

Run this ONCE before starting the training loop:

    python -m ml_pipeline.precompute_bert \\
        --csv      data/manifest.csv \\
        --text_col clinical_notes \\
        --out_dir  data/bert_embs/ \\
        --batch    32

Then add the generated filename to the `bert_filename` column of the manifest.

Requirements:
    pip install transformers pandas tqdm torch

Notes:
    • The script runs on CUDA (NVIDIA) → MPS (Apple Silicon) → CPU fallback.
    • ClinicalBERT produces 768-dim [CLS] embeddings.
    • Max token length is 512 (BERT limit). Longer notes are truncated.
    • Embeddings are L2-normalised before saving (matching the loss.py
      projection head's assumption of pre-normalised inputs).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQ_LEN = 512


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(
    texts:     list[str],
    model:     AutoModel,
    tokenizer: AutoTokenizer,
    device:    torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Encode a list of clinical note strings into (N, 768) CLS embeddings.

    Processes texts in mini-batches to avoid OOM on long datasets.
    """
    all_embs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        out = model(**enc)
        # [CLS] token is at position 0 of the last hidden state
        cls_emb = out.last_hidden_state[:, 0, :]   # (batch, 768)
        cls_emb = F.normalize(cls_emb, dim=-1)      # L2 normalise
        all_embs.append(cls_emb.cpu())

    return torch.cat(all_embs, dim=0)               # (N, 768)


def precompute(
    csv_path:   Path,
    text_col:   str,
    out_dir:    Path,
    batch_size: int = 32,
    overwrite:  bool = False,
) -> None:
    """
    Main driver: reads manifest CSV, encodes notes, saves .pt files.

    The `bert_filename` column is added/updated in-place in the CSV.
    """
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device: CUDA → MPS → CPU ─────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[precompute_bert] Using device: {device}")

    # ── Load ClinicalBERT (frozen) ────────────────────────────────────────
    print(f"[precompute_bert] Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[precompute_bert] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Determine output filenames ────────────────────────────────────────
    def make_bert_fname(row) -> str:
        stem = Path(str(row["image_filename"])).stem
        return f"{stem}_bert.pt"

    df["bert_filename"] = df.apply(make_bert_fname, axis=1)

    # ── Skip already-computed rows ────────────────────────────────────────
    if not overwrite:
        pending_mask = df["bert_filename"].apply(
            lambda fn: not (out_dir / fn).exists()
        )
        pending_df = df[pending_mask]
        print(
            f"[precompute_bert] {len(pending_df)} / {len(df)} rows need encoding."
        )
    else:
        pending_df = df
        print(f"[precompute_bert] Overwrite=True. Encoding all {len(df)} rows.")

    if len(pending_df) == 0:
        print("[precompute_bert] All embeddings already exist. Done.")
        df.to_csv(csv_path, index=False)
        return

    # ── Encode in batches ─────────────────────────────────────────────────
    texts    = pending_df[text_col].fillna("").tolist()
    filenames = pending_df["bert_filename"].tolist()

    print(f"[precompute_bert] Encoding {len(texts)} texts in batches of {batch_size}...")
    with tqdm(total=len(texts), unit="text", desc="ClinicalBERT") as pbar:
        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start : start + batch_size]
            chunk_files = filenames[start : start + batch_size]

            enc = tokenizer(
                chunk_texts,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model(**enc)
                cls_emb = out.last_hidden_state[:, 0, :]   # (B, 768)
                cls_emb = F.normalize(cls_emb, dim=-1)
                cls_emb = cls_emb.cpu()

            for emb, fname in zip(cls_emb, chunk_files):
                torch.save(emb, out_dir / fname)           # saves FloatTensor (768,)

            pbar.update(len(chunk_texts))

    # ── Write updated manifest (bert_filename column added) ───────────────
    df.to_csv(csv_path, index=False)
    print(f"[precompute_bert] Saved {len(pending_df)} embeddings to: {out_dir}")
    print(f"[precompute_bert] Updated manifest written to: {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute ClinicalBERT embeddings for the JEPA training set."
    )
    parser.add_argument("--csv",      type=Path, required=True,  help="Path to manifest CSV.")
    parser.add_argument("--text_col", type=str,  default="clinical_notes",
                        help="Name of the column containing clinical text (default: clinical_notes).")
    parser.add_argument("--out_dir",  type=Path, required=True,  help="Directory to save .pt files.")
    parser.add_argument("--batch",    type=int,  default=32,     help="Tokeniser batch size.")
    parser.add_argument("--overwrite",action="store_true",       help="Re-encode already-saved files.")
    args = parser.parse_args()

    precompute(
        csv_path=args.csv,
        text_col=args.text_col,
        out_dir=args.out_dir,
        batch_size=args.batch,
        overwrite=args.overwrite,
    )
