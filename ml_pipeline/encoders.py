"""
ml_pipeline/encoders.py
=======================
V-JEPA Dual-Encoder Module
--------------------------
Implements:
    • ContextEncoder  — a Vision Transformer (ViT) that encodes visible/contextual
                        patches of a 2-D medical scan image.
    • TargetEncoder   — an identical ViT architecture whose weights are updated via
                        Exponential Moving Average (EMA) of the ContextEncoder weights.
                        It is never directly optimised by gradient descent.

Design references:
    Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
    Predictive Architecture", CVPR 2023 (I-JEPA).
    Extended here for volumetric / report-image inputs.

Usage:
    >>> from ml_pipeline.encoders import ContextEncoder, TargetEncoder, build_encoder_pair
    >>> ctx_enc, tgt_enc = build_encoder_pair(cfg)
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EncoderConfig:
    """
    Hyper-parameters for both the Context and Target ViT encoders.

    Attributes
    ----------
    image_size : int
        Spatial resolution of the input image (assumed square).
        Default: 224  (standard for ViT-B).
    patch_size : int
        Edge length of each patch token (pixels). Must evenly divide ``image_size``.
        Default: 16  → 196 tokens for a 224×224 image.
    in_channels : int
        Number of input image channels (e.g. 1 for greyscale X-ray, 3 for RGB).
        Default: 3.
    embed_dim : int
        Dimensionality of the patch embedding / transformer hidden size (d_model).
        Default: 768  (ViT-Base).
    depth : int
        Number of transformer encoder blocks.
        Default: 12  (ViT-Base).
    num_heads : int
        Number of multi-head attention heads.
        Default: 12  (embed_dim must be divisible by num_heads).
    mlp_ratio : float
        Expansion ratio inside each transformer FFN block.
        Default: 4.0  → inner dim = embed_dim * mlp_ratio.
    dropout : float
        Dropout probability applied to attention weights and FFN activations.
        Default: 0.1.
    emb_dropout : float
        Dropout probability applied after patch + positional embedding sum.
        Default: 0.0.
    ema_decay : float
        Momentum for target-encoder EMA update:  θ_t ← τ·θ_t + (1-τ)·θ_c.
        Typical range: [0.996, 0.9999].
        Default: 0.998.
    use_cls_token : bool
        Whether to prepend a learnable [CLS] token.
        When True the output at position 0 is the global representation;
        the remaining positions are patch tokens.
        Default: True.
    """
    image_size:    int   = 224
    patch_size:    int   = 16
    in_channels:   int   = 3
    embed_dim:     int   = 768
    depth:         int   = 12
    num_heads:     int   = 12
    mlp_ratio:     float = 4.0
    dropout:       float = 0.1
    emb_dropout:   float = 0.0
    ema_decay:     float = 0.998
    use_cls_token: bool  = True

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def num_patches(self) -> int:
        assert self.image_size % self.patch_size == 0, (
            f"image_size ({self.image_size}) must be divisible by "
            f"patch_size ({self.patch_size})."
        )
        return (self.image_size // self.patch_size) ** 2

    @property
    def seq_len(self) -> int:
        """Full sequence length fed into the Transformer (including CLS if used)."""
        return self.num_patches + (1 if self.use_cls_token else 0)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Building Blocks
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits a 2-D image into non-overlapping patches and linearly projects
    each flattened patch to ``embed_dim`` dimensions.

    Input  : (B, C, H, W)
    Output : (B, num_patches, embed_dim)
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=False,   # bias handled by LayerNorm later
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        x = self.proj(x)             # → (B, embed_dim, H/P, W/P)
        x = x.flatten(2)             # → (B, embed_dim, num_patches)
        x = x.transpose(1, 2)        # → (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed 2-D sinusoidal positional encoding for grid-arranged patch tokens.
    Compatible with arbitrary sequence lengths up to ``max_len``.

    Output shape : (1, seq_len, embed_dim)  — broadcastable over batch.
    """

    def __init__(self, cfg: EncoderConfig, max_len: int = 4096) -> None:
        super().__init__()
        d = cfg.embed_dim
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter — not updated by optimiser)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape (B, seq_len, d)."""
        return x + self.pe[:, : x.size(1), :]


class MLP(nn.Module):
    """Two-layer Feed-Forward Network used inside each Transformer block."""

    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        inner_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer encoder block.

    Pre-LN is now preferred over original Post-LN for training stability,
    especially at smaller batch sizes common in medical imaging.

    Sub-layers:
        1. LayerNorm → MultiHeadSelfAttention → residual
        2. LayerNorm → MLP                   → residual
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.attn  = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,   # expects (B, T, D) — PyTorch ≥ 1.9
        )
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.mlp   = MLP(cfg.embed_dim, cfg.mlp_ratio, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, T, D)
        key_padding_mask : BoolTensor  (B, T), optional
            Positions set to True are **ignored** by attention.
            Used to mask out prediction-target tokens from context.

        Returns
        -------
        x : Tensor  (B, T, D)
        """
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Vision Transformer Backbone
# ──────────────────────────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    A generic ViT backbone shared by both Context and Target encoders.

    Architecture
    ------------
    PatchEmbedding  →  [CLS token]  →  PositionalEncoding  →
    Dropout  →  N × TransformerBlock  →  LayerNorm  →  output

    Output shape
    ------------
    (B, seq_len, embed_dim)   where seq_len = num_patches + int(use_cls_token)
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Patch embedding ──────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(cfg)

        # ── Optional learnable CLS token ─────────────────────────────────────
        self.cls_token: Optional[nn.Parameter] = None
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(cfg)

        # ── Embedding dropout ────────────────────────────────────────────────
        self.emb_drop = nn.Dropout(p=cfg.emb_dropout)

        # ── Transformer stack ────────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.depth)]
        )

        # ── Output normalisation ─────────────────────────────────────────────
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

        self._init_weights()

    # ── Weight initialisation ───────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Apply truncated normal init to linear layers and patch projection."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor  (B, C, H, W)
            Batch of input images (or single-channel scans).
        mask : BoolTensor  (B, num_patches), optional
            Token-level mask passed as key_padding_mask to each attention block.
            True  → masked out (context encoder: visible patches only).
            False → attended to.

        Returns
        -------
        tokens : FloatTensor  (B, seq_len, embed_dim)
        """
        B = x.size(0)

        # 1. Patch embedding
        tokens = self.patch_embed(x)                  # (B, N, D)

        # 2. Prepend CLS token
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)

        # 3. Add positional encoding
        tokens = self.pos_enc(tokens)

        # 4. Embedding dropout
        tokens = self.emb_drop(tokens)

        # 5. Adjust mask for CLS token position (CLS is never masked)
        key_padding_mask = None
        if mask is not None:
            if self.cls_token is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
                key_padding_mask = torch.cat([cls_mask, mask], dim=1)  # (B, N+1)
            else:
                key_padding_mask = mask  # (B, N)

        # 6. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)

        # 7. Output LayerNorm
        tokens = self.norm(tokens)
        return tokens


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Context Encoder  (gradient-updated)
# ──────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    The *online* encoder in V-JEPA.

    Receives a masked view of the image (visible patches only) and produces
    latent representations that the Predictor will use to reconstruct the
    target encoder's representations at masked positions.

    The ContextEncoder is directly optimised by the training loss.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vit  = VisionTransformer(cfg)

    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor  (B, C, H, W)
        context_mask : BoolTensor  (B, num_patches), optional
            True at positions that are *masked out* (not visible).
            When None, all patches are treated as visible.

        Returns
        -------
        FloatTensor  (B, seq_len, embed_dim)
        """
        return self.vit(x, mask=context_mask)

    def get_patch_tokens(self, x: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method: returns only the patch token representations,
        stripping the CLS token if present.

        Returns
        -------
        FloatTensor  (B, num_patches, embed_dim)
        """
        tokens = self.forward(x, context_mask)
        if self.cfg.use_cls_token:
            tokens = tokens[:, 1:, :]   # drop CLS
        return tokens


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Target Encoder  (EMA-updated, no gradient)
# ──────────────────────────────────────────────────────────────────────────────

class TargetEncoder(nn.Module):
    """
    The *momentum* encoder in V-JEPA.

    Its weights start as a copy of the ContextEncoder and are updated each
    training step via Exponential Moving Average (EMA):

        θ_target  ←  τ · θ_target  +  (1 − τ) · θ_context

    where τ = ``cfg.ema_decay`` (e.g. 0.998).

    The TargetEncoder receives the *full* unmasked image and outputs the
    ground-truth latent representations that the Predictor must match.

    No gradients flow through the TargetEncoder.
    """

    def __init__(self, cfg: EncoderConfig, context_encoder: ContextEncoder) -> None:
        super().__init__()
        self.cfg   = cfg
        self.decay = cfg.ema_decay

        # Deep copy the context encoder's ViT — shares architecture, not weights
        self.vit = copy.deepcopy(context_encoder.vit)

        # Freeze all parameters — EMA update is applied manually
        for p in self.vit.parameters():
            p.requires_grad = False

    # ── EMA Update ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def update(self, context_encoder: ContextEncoder) -> None:
        """
        Perform one EMA step.

        Call this once per optimiser step, *after* the context_encoder's
        parameters have been updated by the optimiser.

        Example
        -------
        >>> optimizer.step()
        >>> target_encoder.update(context_encoder)
        """
        for param_t, param_c in zip(
            self.vit.parameters(), context_encoder.vit.parameters()
        ):
            param_t.data.mul_(self.decay).add_(
                param_c.data, alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def update_decay(self, new_decay: float) -> None:
        """
        Optionally anneal the EMA decay during training (cosine schedule).

        A common practice is to ramp τ from 0.996 → 0.9999 over training.
        """
        self.decay = new_decay

    # ── Forward ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the *full* unmasked image.  No mask is applied.

        Parameters
        ----------
        x : FloatTensor  (B, C, H, W)

        Returns
        -------
        FloatTensor  (B, seq_len, embed_dim)
        """
        return self.vit(x, mask=None)

    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns only patch token representations (CLS token stripped).

        Returns
        -------
        FloatTensor  (B, num_patches, embed_dim)
        """
        tokens = self.forward(x)
        if self.cfg.use_cls_token:
            tokens = tokens[:, 1:, :]
        return tokens


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Factory Helper
# ──────────────────────────────────────────────────────────────────────────────

def build_encoder_pair(
    cfg: Optional[EncoderConfig] = None,
) -> Tuple[ContextEncoder, TargetEncoder]:
    """
    Instantiate a matched (ContextEncoder, TargetEncoder) pair.

    The TargetEncoder is initialised with weights cloned from the
    ContextEncoder.

    Parameters
    ----------
    cfg : EncoderConfig, optional
        If None, uses default values (ViT-Base, 224px, patch 16).

    Returns
    -------
    ctx_enc : ContextEncoder
    tgt_enc : TargetEncoder

    Example
    -------
    >>> cfg = EncoderConfig(image_size=224, patch_size=16, embed_dim=768)
    >>> ctx_enc, tgt_enc = build_encoder_pair(cfg)
    >>> x = torch.randn(2, 3, 224, 224)
    >>> print(ctx_enc(x).shape)      # (2, 197, 768)
    >>> print(tgt_enc(x).shape)      # (2, 197, 768)
    """
    if cfg is None:
        cfg = EncoderConfig()

    ctx_enc = ContextEncoder(cfg)
    tgt_enc = TargetEncoder(cfg, ctx_enc)
    return ctx_enc, tgt_enc


# ──────────────────────────────────────────────────────────────────────────────
# 6.  EMA Decay Schedule  (utility, used in train.py)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_ema_schedule(
    step: int,
    total_steps: int,
    start_decay: float = 0.996,
    end_decay:   float = 0.9999,
) -> float:
    """
    Cosine ramp for the EMA decay τ from ``start_decay`` → ``end_decay``.

    Increasing τ over training prevents the target from collapsing early on
    while providing a stable, slowly-moving target in later stages.

    Parameters
    ----------
    step         : Current global training step (0-indexed).
    total_steps  : Total number of training steps.
    start_decay  : τ at step 0.
    end_decay    : τ at step total_steps.

    Returns
    -------
    float  — the EMA decay value for the current step.
    """
    progress = step / max(total_steps - 1, 1)
    # cosine annealing: goes from 0 → 1 over progress 0 → 1
    cosine_val = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_decay + (end_decay - start_decay) * cosine_val


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Quick Smoke Test  (run: python -m ml_pipeline.encoders)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  V-JEPA Encoder Smoke Test")
    print("=" * 60)

    cfg = EncoderConfig(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ema_decay=0.998,
        use_cls_token=True,
    )

    print(f"\nConfig:\n  image_size  : {cfg.image_size}")
    print(f"  patch_size  : {cfg.patch_size}")
    print(f"  num_patches : {cfg.num_patches}")
    print(f"  seq_len     : {cfg.seq_len}  (patches + CLS)")
    print(f"  embed_dim   : {cfg.embed_dim}")
    print(f"  depth       : {cfg.depth}")

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"\nDevice: {device}")

    ctx_enc, tgt_enc = build_encoder_pair(cfg)
    ctx_enc = ctx_enc.to(device)
    tgt_enc = tgt_enc.to(device)

    B = 2
    x = torch.randn(B, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)

    # Context forward (with random mask — 50 % of patches masked)
    mask = torch.rand(B, cfg.num_patches, device=device) > 0.5   # BoolTensor

    ctx_out   = ctx_enc(x, context_mask=mask)
    patch_ctx = ctx_enc.get_patch_tokens(x, mask)

    # Target forward (no mask)
    tgt_out   = tgt_enc(x)
    patch_tgt = tgt_enc.get_patch_tokens(x)

    print(f"\nContext encoder output : {ctx_out.shape}")   # (2, 197, 768)
    print(f"Context patch tokens   : {patch_ctx.shape}")  # (2, 196, 768)
    print(f"Target encoder output  : {tgt_out.shape}")    # (2, 197, 768)
    print(f"Target patch tokens    : {patch_tgt.shape}")  # (2, 196, 768)

    # EMA update
    tgt_enc.update(ctx_enc)
    print("\nEMA update: OK")

    # Cosine decay schedule
    decay_mid = cosine_ema_schedule(step=500, total_steps=1000)
    print(f"EMA decay at step 500 / 1000 : {decay_mid:.6f}")

    # Parameter counts
    ctx_params = sum(p.numel() for p in ctx_enc.parameters() if p.requires_grad)
    tgt_params = sum(p.numel() for p in tgt_enc.parameters())
    print(f"\nContextEncoder trainable params : {ctx_params:,}")
    print(f"TargetEncoder  total     params : {tgt_params:,}  (no grad)")
    print("\n[PASS] All checks passed.")
