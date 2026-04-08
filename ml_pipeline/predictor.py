"""
ml_pipeline/predictor.py
========================
V-JEPA Lightweight Predictor
-----------------------------
Receives:
    • Context tokens  — latent representations of VISIBLE patches
                        output by the ContextEncoder.
    • Masked positions — integer indices pointing to WHERE the
                        missing patches should go in the full sequence.

Predicts:
    • Target tokens   — must match what the TargetEncoder would produce
                        for those masked positions (the loss in loss.py
                        computes MSE between these two in latent space).

Design choices (per user spec):
    • Option A masking — positionally-aware: shared learned [MASK] token
      PLUS the sinusoidal position embedding of each missing position.
    • Option B depth   — 6 layers, 12 heads, embed_dim 768;
      asymmetric with the 12-layer ContextEncoder.
    • MPS-first        — no CUDA-only calls; works on Apple Silicon.

Usage:
    >>> from ml_pipeline.predictor import JEPAPredictor, build_predictor
    >>> predictor = build_predictor(embed_dim=768)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Sinusoidal Position Encoding  (shared utility, mirrors encoders.py)
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal table indexed by absolute patch position.
    Shape: (1, max_len, embed_dim) — register as buffer (not trainable).
    """

    def __init__(self, embed_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Gather positional embeddings for the given absolute indices.

        Parameters
        ----------
        indices : LongTensor  (B, K)
            Absolute patch positions to look up.

        Returns
        -------
        FloatTensor  (B, K, embed_dim)
        """
        # pe: (1, max_len, D)  →  index across the seq dimension
        # Expand to (B, max_len, D) via broadcasting, then gather
        B, K = indices.shape
        pe_expanded = self.pe.expand(B, -1, -1)         # (B, max_len, D)
        idx = indices.unsqueeze(-1).expand(B, K, pe_expanded.size(-1))  # (B, K, D)
        return torch.gather(pe_expanded, dim=1, index=idx)              # (B, K, D)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  JEPAPredictor
# ──────────────────────────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """
    Lightweight Predictor Network for V-JEPA.

    Configuration (Option B):
        depth     = 6 layers
        num_heads = 12
        embed_dim = 768
        FFN width = embed_dim × 4 = 3072

    Masking strategy (Option A — positionally-aware):
        Each masked position receives a *shared* learnable [MASK] token
        PLUS the sinusoidal positional embedding of that specific patch index.
        The predictor therefore knows both "something is missing here" and
        "this missing thing lives at position k in the full image grid."

    Forward contract:
        Input  — (context_tokens, masked_positions)
        Output — predicted_tokens  (B, num_masked, embed_dim)
                 These should match the TargetEncoder's output for those positions.
    """

    def __init__(
        self,
        embed_dim:   int = 768,
        num_heads:   int = 12,
        depth:       int = 6,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # ── Shared learnable [MASK] embedding ────────────────────────────────
        # A single vector broadcast to every masked position before positional
        # encoding is added. Initialised with small truncated-normal values.
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ── Sinusoidal position table ─────────────────────────────────────────
        # Shared with encoder positions — same fixed table, just looked up
        # differently (gather by index vs. slice).
        self.pos_enc = SinusoidalPE(embed_dim, max_len=max_seq_len)

        # ── Lightweight Transformer stack (Pre-LN, GELU) ─────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, T, D)
            norm_first=True,    # Pre-LN — stable for JEPA asymmetric training
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # ── Output layer norm ────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(
        self,
        context_tokens:   torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context_tokens : FloatTensor  (B, num_context, embed_dim)
            Patch latents for visible positions, from ContextEncoder.
            These already carry positional information baked in during encoding.

        masked_positions : LongTensor  (B, num_masked)
            Absolute patch indices of the holes to predict.
            Values must be in [0, max_seq_len).

        Returns
        -------
        predicted_tokens : FloatTensor  (B, num_masked, embed_dim)
            Predicted latent representations for each masked patch.
            Compared against TargetEncoder output by the MSE loss.

        Sequence layout passed to the Transformer:
            [ context_tokens (num_context) | mask_tokens (num_masked) ]
                                           ^
                                    predictions extracted here
        """
        B, _, _ = context_tokens.shape
        num_masked = masked_positions.shape[1]

        # ── 1. Build positionally-aware mask tokens ─────────────────────────
        #   Expand shared mask embedding  →  (B, num_masked, D)
        mask_tokens = self.mask_token.expand(B, num_masked, -1).clone()

        #   Gather positional encodings for the masked patch indices (vectorised)
        pos_embs = self.pos_enc.lookup(masked_positions)     # (B, num_masked, D)
        mask_tokens = mask_tokens + pos_embs

        # ── 2. Concatenate: [context | masks] ──────────────────────────────
        x = torch.cat([context_tokens, mask_tokens], dim=1)  # (B, C+M, D)

        # ── 3. Lightweight Transformer ──────────────────────────────────────
        x = self.blocks(x)
        x = self.norm(x)

        # ── 4. Extract only the predicted positions (tail of sequence) ──────
        predicted_tokens = x[:, -num_masked:, :]             # (B, M, D)
        return predicted_tokens


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_predictor(
    embed_dim:   int = 768,
    num_heads:   int = 12,
    depth:       int = 6,
    dropout:     float = 0.1,
    max_seq_len: int = 1024,
) -> JEPAPredictor:
    """Instantiate a predictor with Option-B defaults."""
    return JEPAPredictor(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Smoke Test  (python -m ml_pipeline.predictor)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  V-JEPA Predictor Smoke Test")
    print("=" * 60)

    # ── Device selection: CUDA → MPS → CPU ──────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("[WARN] No GPU found — running on CPU.")
    print(f"\nDevice: {device}")

    # ── Model ───────────────────────────────────────────────────────────────
    predictor = build_predictor().to(device)

    total_params = sum(p.numel() for p in predictor.parameters())
    trainable    = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"\nJEPAPredictor  —  depth=6, heads=12, dim=768")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    # ── Forward pass ────────────────────────────────────────────────────────
    B            = 2
    num_context  = 100   # visible patches fed from ContextEncoder
    num_masked   = 30    # patches to predict  (≈ 23 % of 128-patch sequence)
    embed_dim    = 768

    context_tokens   = torch.randn(B, num_context, embed_dim, device=device)
    masked_positions = torch.randint(0, 196, (B, num_masked), device=device)

    predictions = predictor(context_tokens, masked_positions)
    print(f"\nContext tokens shape  : {context_tokens.shape}")
    print(f"Masked positions shape: {masked_positions.shape}")
    print(f"Predicted tokens shape: {predictions.shape}")
    assert predictions.shape == (B, num_masked, embed_dim), "Shape mismatch!"
    print("\n[PASS] Output shape is correct: (2, 30, 768)")
    print("[PASS] All checks passed.")
