"""
ml_pipeline/train.py
=====================
Tabular Imputation Training Loop — MedFill VQ-VAE
---------------------------------------------------

Trains a Transformer-based tabular imputation model on the combined
Anemia + CKD dataset. Given a row with some lab values masked, the model
learns to reconstruct ALL features — including the missing ones.

This is the core ML model that fills in missing biomarkers (Hb, MCV,
Creatinine, etc.) from partial lab reports extracted by the AI agents.

Architecture:
    masked_x  (B, F)
        ↓  linear embed per feature
    Transformer Encoder  (3 layers, 4 heads)
        ↓
    Linear projection head
    reconstructed_x  (B, F)

Loss:
    MSE on the originally-OBSERVED positions only (the model must not
    just memorise the masked zeros — it must predict real values).

Quick start:
    python -m ml_pipeline.train

Resume from checkpoint:
    python -m ml_pipeline.train --resume ml_pipeline/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.data.dataset import build_loaders, NUM_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"[train] Device: {dev}  ({props.name}, {props.total_memory // 1024**2} MB VRAM)")
    else:
        dev = torch.device("cpu")
        print("[train] Device: cpu  (no GPU found)")
    return dev


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model — Tabular Transformer Imputer
# ──────────────────────────────────────────────────────────────────────────────

class TabularImputerModel(nn.Module):
    """
    Lightweight Transformer that reconstructs masked tabular features.

    Each of the F features is treated as a separate token (like a word).
    The model sees all tokens (masked ones have value 0) and predicts
    the correct value for every position.

    Parameters
    ----------
    num_features : int    Number of input features (columns). Default: 25.
    embed_dim    : int    Per-feature embedding size. Default: 64.
    depth        : int    Number of Transformer encoder layers. Default: 3.
    num_heads    : int    Attention heads. Default: 4.
    dropout      : float  Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        num_features: int   = NUM_FEATURES,
        embed_dim:    int   = 64,
        depth:        int   = 3,
        num_heads:    int   = 4,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.embed_dim    = embed_dim

        # Each feature is projected from scalar → embed_dim
        self.feature_embed = nn.Linear(1, embed_dim)

        # Learnable position embedding — one per feature column
        self.pos_embed = nn.Embedding(num_features, embed_dim)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Per-feature output head: embed_dim → scalar
        self.output_head = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor (B, F)   masked feature vector (missing → 0.0)

        Returns
        -------
        FloatTensor (B, F)       reconstructed feature vector
        """
        B, F = x.shape

        # (B, F, 1) → (B, F, embed_dim)
        tokens = self.feature_embed(x.unsqueeze(-1))

        # Add positional encoding per feature index
        positions = torch.arange(F, device=x.device).unsqueeze(0)  # (1, F)
        tokens = tokens + self.pos_embed(positions)                 # (B, F, D)

        # Transformer
        tokens = self.transformer(tokens)                           # (B, F, D)

        # Project each token back to a scalar
        out = self.output_head(tokens).squeeze(-1)                  # (B, F)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Loss — MSE on observed positions only
# ──────────────────────────────────────────────────────────────────────────────

def imputation_loss(
    pred:     torch.Tensor,   # (B, F) model output
    target:   torch.Tensor,   # (B, F) original (unmasked) values
    obs_mask: torch.Tensor,   # (B, F) 1=originally observed, 0=was NaN in data
    feature_weights: torch.Tensor = None,  # (F,) optional per-feature weight
) -> torch.Tensor:
    """
    Weighted MSE on positions that were originally present in the dataset.
    NHANES rows are ~60% sparse (NaN for CKD categoricals), so features
    that ARE observed in NHANES carry more signal and get upweighted.
    We don't penalise the model for positions that were ALWAYS missing.
    """
    diff  = (pred - target) ** 2          # (B, F)
    valid = obs_mask.bool()               # where ground-truth exists
    if valid.sum() == 0:
        return diff.mean()                # fallback: shouldn't happen

    if feature_weights is not None:
        # (B, F) weight per feature, broadcast
        w = feature_weights.unsqueeze(0).expand_as(diff)  # (B, F)
        weighted = diff * w
        return weighted[valid].mean()
    return diff[valid].mean()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
    }, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[int, float]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"[train] Resumed from epoch {state['epoch']} (best_val={state['best_val']:.5f})")
    return state["epoch"] + 1, state["best_val"]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  One epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:     TabularImputerModel,
    loader:    torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    is_train:  bool = True,
) -> float:
    model.train(is_train)
    total_loss = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            masked_x   = batch["masked_x"].to(device)    # (B, F)  — input
            original_x = batch["original_x"].to(device)  # (B, F)  — target
            obs_mask   = batch["obs_mask"].to(device)     # (B, F)  — valid mask

            pred = model(masked_x)
            loss = imputation_loss(pred, original_x, obs_mask)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main training driver
# ──────────────────────────────────────────────────────────────────────────────

def train(
    data_dir:    str   = "data/raw",
    epochs:      int   = 50,
    batch_size:  int   = 64,
    lr:          float = 3e-4,
    embed_dim:   int   = 64,
    depth:       int   = 3,
    num_heads:   int   = 4,
    dropout:     float = 0.1,
    mask_ratio:  float = 0.30,
    ckpt_dir:    str   = "ml_pipeline/checkpoints",
    resume_path: Optional[str] = None,
    seed:        int   = 42,
) -> None:

    set_seed(seed)
    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    print("[train] Building dataloaders...")
    train_loader, val_loader, meta = build_loaders(
        raw_dir    = data_dir,
        batch_size = batch_size,
        mask_ratio = mask_ratio,
        seed       = seed,
    )
    F = meta["num_features"]
    print(f"[train] Features={F} | Train={meta['train_size']} | Val={meta['val_size']}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TabularImputerModel(
        num_features = F,
        embed_dim    = embed_dim,
        depth        = depth,
        num_heads    = num_heads,
        dropout      = dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model params: {total_params:,}")

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup for 5% of total steps, then cosine decay
    total_steps  = epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    if resume_path and Path(resume_path).exists():
        state = torch.load(Path(resume_path), map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch  = state["epoch"] + 1
        best_val     = state["best_val"]
        global_step  = state.get("global_step", 0)
        print(f"[train] Resumed from epoch {state['epoch']} (best_val={best_val:.5f})")

    # -- Feature-presence weights: up-weight features rare in NHANES --------
    # Compute obs rate per feature across the training set (first pass)
    print("[train] Computing per-feature observation rates...")
    feat_obs_sum   = torch.zeros(F, device=device)
    feat_total     = 0
    with torch.no_grad():
        for batch in train_loader:
            om = batch["obs_mask"].to(device)   # (B, F)
            feat_obs_sum += om.sum(0)
            feat_total   += om.shape[0]
    obs_rate = feat_obs_sum / max(feat_total, 1)    # (F,) in [0, 1]
    # Inverse-frequency weight, clipped to [0.5, 4.0]
    feat_weights = (1.0 / (obs_rate + 1e-6)).clamp(0.5, 4.0)
    feat_weights = feat_weights / feat_weights.mean()   # normalise to mean=1
    print(f"[train] Feature weights: min={feat_weights.min():.2f}  "
          f"max={feat_weights.max():.2f}  mean={feat_weights.mean():.2f}")

    ckpt_path = Path(ckpt_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MedFill Tabular Imputation Training")
    print(f"  Epochs={epochs} | LR={lr} | BatchSize={batch_size} | MaskRatio={mask_ratio}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── train ──
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            masked_x   = batch["masked_x"].to(device)
            original_x = batch["original_x"].to(device)
            obs_mask   = batch["obs_mask"].to(device)

            pred = model(masked_x)
            loss = imputation_loss(pred, original_x, obs_mask, feat_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()
            n_train    += 1
        train_loss /= max(n_train, 1)

        # ── val ──
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                masked_x   = batch["masked_x"].to(device)
                original_x = batch["original_x"].to(device)
                obs_mask   = batch["obs_mask"].to(device)
                pred = model(masked_x)
                loss = imputation_loss(pred, original_x, obs_mask, feat_weights)
                val_loss += loss.item()
                n_val    += 1
        val_loss /= max(n_val, 1)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:>3}/{epochs}]  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"lr={lr_now:.2e}  ({elapsed:.1f}s)"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "global_step": global_step,
            }, ckpt_path / "best_model.pt")
            print(f"  [best] val={best_val:.5f}  -> saved")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "global_step": global_step,
            }, ckpt_path / f"epoch_{epoch+1:04d}.pt")

    print(f"\n{'='*60}")
    print(f"  Training Complete!  Best val loss: {best_val:.5f}")
    print(f"  Checkpoint: {(ckpt_path / 'best_model.pt').resolve()}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFill Tabular Imputation Training")
    parser.add_argument("--data-dir",   default="data/raw",                        help="Raw CSV directory")
    parser.add_argument("--epochs",     default=80,   type=int,                    help="Number of epochs")
    parser.add_argument("--batch-size", default=128,  type=int,                    help="Batch size")
    parser.add_argument("--lr",         default=3e-4, type=float,                  help="Learning rate")
    parser.add_argument("--embed-dim",  default=128,  type=int,                    help="Feature embedding dim")
    parser.add_argument("--depth",      default=4,    type=int,                    help="Transformer layers")
    parser.add_argument("--mask-ratio", default=0.35, type=float,                  help="Fraction of features to mask during training")
    parser.add_argument("--ckpt-dir",   default="ml_pipeline/checkpoints",         help="Checkpoint output directory")
    parser.add_argument("--resume",     default=None,                              help="Path to checkpoint to resume from")
    parser.add_argument("--seed",       default=42,   type=int,                    help="Random seed")
    args = parser.parse_args()

    train(
        data_dir    = args.data_dir,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        embed_dim   = args.embed_dim,
        depth       = args.depth,
        mask_ratio  = args.mask_ratio,
        ckpt_dir    = args.ckpt_dir,
        resume_path = args.resume,
        seed        = args.seed,
    )
