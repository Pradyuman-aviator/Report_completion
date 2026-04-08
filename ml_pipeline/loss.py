"""
ml_pipeline/loss.py
===================
Composite Loss for the Medical V-JEPA + ClinicalBERT Distillation Pipeline
---------------------------------------------------------------------------

Two distinct objectives are combined here:

  L_total = λ_jepa · L_jepa  +  λ_distil · L_distil

  ┌─────────────────────────────────────────────────────────────────────┐
  │  L_jepa   — MSE in latent space                                     │
  │             Predictor output  vs.  TargetEncoder output             │
  │             (the core self-supervised signal, no labels needed)     │
  ├─────────────────────────────────────────────────────────────────────┤
  │  L_distil — Contrastive alignment loss (NT-Xent style)              │
  │             Vision patch embeddings  vs.  ClinicalBERT text embeds  │
  │             (grounding vision representations in clinical language)  │
  └─────────────────────────────────────────────────────────────────────┘

Design decisions:
  • Smooth-L1 is offered as an alternative to MSE for L_jepa — it is
    less sensitive to outlier patches (e.g. artefact-heavy X-ray regions).
  • NT-Xent (Normalised Temperature-scaled Cross Entropy) is used for
    distillation — it naturally pulls vision/text embeddings of the
    same report together while pushing apart different reports in the batch.
  • A linear projection head bridges the dimension gap between the ViT
    embed space and ClinicalBERT's 768-d output (they may already match
    if embed_dim=768, but the head allows independent scaling/dropout).
  • All ops are MPS-compatible (no CUDA-only calls).

References:
  Chen et al., "A Simple Framework for Contrastive Learning" (SimCLR), 2020.
  Alsentzer et al., "Publicly Available Clinical BERT Embeddings", 2019.
  Assran et al., "Self-Supervised Learning from Images with a JEPA", 2023.

Usage:
    >>> from ml_pipeline.loss import VJEPALoss, ClinicalDistilLoss, CombinedLoss
    >>> criterion = CombinedLoss(lambda_jepa=1.0, lambda_distil=0.5)
    >>> loss, info = criterion(predicted, target, vis_emb, txt_emb)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Loss Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LossConfig:
    """
    Hyper-parameters for the composite loss function.

    Attributes
    ----------
    lambda_jepa : float
        Weight for the V-JEPA latent MSE loss term. Default: 1.0.
    lambda_distil : float
        Weight for the ClinicalBERT contrastive distillation loss. Default: 0.5.
        Set to 0.0 to disable distillation (e.g., during pretraining without text).
    jepa_loss_type : str
        'mse'      — Mean Squared Error (original V-JEPA).
        'smooth_l1'— Huber / Smooth-L1 (robust to outlier patches in scans).
    smooth_l1_beta : float
        β for Smooth-L1 loss. Ignored if jepa_loss_type='mse'. Default: 1.0.
    temperature : float
        Temperature τ for NT-Xent contrastive loss. Typical range: [0.05, 0.2].
        Default: 0.07.
    proj_vis_dim : int
        Input dim of the vision projection head (= encoder embed_dim). Default: 768.
    proj_txt_dim : int
        Input dim of the text projection head (= ClinicalBERT hidden size). Default: 768.
    proj_out_dim : int
        Shared output dim for both projection heads. Default: 256.
    proj_dropout : float
        Dropout inside projection heads. Default: 0.1.
    normalize_jepa : bool
        If True, L2-normalise both predicted and target vectors before MSE.
        This prevents the loss from collapsing to zero by shrinking norms.
        Default: True.
    """
    lambda_jepa:    float = 1.0
    lambda_distil:  float = 0.5
    jepa_loss_type: str   = "mse"           # 'mse' | 'smooth_l1'
    smooth_l1_beta: float = 1.0
    temperature:    float = 0.07
    proj_vis_dim:   int   = 768
    proj_txt_dim:   int   = 768
    proj_out_dim:   int   = 256
    proj_dropout:   float = 0.1
    normalize_jepa: bool  = True


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Projection Head  (shared by vision and text towers)
# ──────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head that maps encoder features to a
    shared latent space for contrastive comparison.

    Architecture:
        Linear(in_dim → in_dim)  →  BatchNorm  →  GELU  →  Dropout
        →  Linear(in_dim → out_dim)

    Following SimCLR v2: a non-linear head improves downstream alignment
    even when both modalities share the same dimensionality.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=False),
        )
        # Initialise the final linear with small weights for training stability
        nn.init.trunc_normal_(self.net[-1].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor  (B, in_dim)  — pooled feature vector per sample.

        Returns
        -------
        FloatTensor  (B, out_dim)   — L2-normalised embedding.
        """
        x = self.net(x)
        return F.normalize(x, dim=-1)   # unit-sphere normalisation


# ──────────────────────────────────────────────────────────────────────────────
# 2.  V-JEPA Latent MSE Loss
# ──────────────────────────────────────────────────────────────────────────────

class VJEPALoss(nn.Module):
    """
    Core self-supervised loss: mean squared error between predicted latents
    and the (stop-gradient) TargetEncoder latents at masked positions.

    L_jepa = mean_{masked patches} ‖ ŷ − y ‖²₂

    where:
        ŷ  = JEPAPredictor(context_tokens, masked_positions)
        y  = TargetEncoder(full_image)[:, masked_positions, :]   ← no grad

    Optional L2 normalisation (normalize_jepa=True) prevents the trivial
    solution of collapsing both vectors to zero.  With normalisation the
    loss measures cosine distance:

        L_jepa = ‖ ŷ/‖ŷ‖  −  y/‖y‖ ‖²₂
               = 2 · (1 − cosine_similarity(ŷ, y))
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg       = cfg
        self.loss_type = cfg.jepa_loss_type
        self.beta      = cfg.smooth_l1_beta
        self.normalize = cfg.normalize_jepa

    def forward(
        self,
        predicted: torch.Tensor,
        target:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predicted : FloatTensor  (B, num_masked, embed_dim)
            Output of JEPAPredictor for the masked positions.
        target : FloatTensor  (B, num_masked, embed_dim)
            Output of TargetEncoder at the same masked positions.
            Must be detached from the computation graph (stop-gradient).

        Returns
        -------
        loss : scalar FloatTensor
        """
        assert predicted.shape == target.shape, (
            f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}"
        )

        if self.normalize:
            predicted = F.normalize(predicted, dim=-1)
            target    = F.normalize(target,    dim=-1)

        if self.loss_type == "mse":
            loss = F.mse_loss(predicted, target, reduction="mean")
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predicted, target, beta=self.beta, reduction="mean")
        else:
            raise ValueError(
                f"Unknown jepa_loss_type '{self.loss_type}'. "
                "Choose 'mse' or 'smooth_l1'."
            )
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ClinicalBERT Distillation Loss  (NT-Xent Contrastive)
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalDistilLoss(nn.Module):
    """
    Contrastive knowledge-distillation loss that aligns the ViT's visual
    representations with ClinicalBERT's text embeddings for the same report.

    Mechanism (NT-Xent, à la SimCLR / CLIP):
    -----------------------------------------
    For a batch of B (image, report) pairs:
      1. Project vision embeddings → z_v  (B, proj_out_dim), L2-normalised.
      2. Project text  embeddings → z_t  (B, proj_out_dim), L2-normalised.
      3. Compute similarity matrix:  S = z_v @ z_t.T / τ   (B × B)
      4. Positive pairs are on the diagonal; all others are negatives.
      5. L_distil = 0.5 · (CE_loss(S, diag) + CE_loss(S.T, diag))
         — symmetric cross-entropy, averaged over both directions.

    Why mutual information maximisation helps here:
      ClinicalBERT encodes structured clinical knowledge from millions of
      EHR notes. Aligning the vision model to this space implicitly injects
      clinical priors (e.g. "hyperdense region" ≡ "haemorrhage") without
      any task-specific labels — a form of soft knowledge distillation.
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.temperature = cfg.temperature

        # Separate projection heads — in_dim may differ if ViT ≠ BERT dim
        self.vision_proj = ProjectionHead(
            in_dim=cfg.proj_vis_dim,
            out_dim=cfg.proj_out_dim,
            dropout=cfg.proj_dropout,
        )
        self.text_proj = ProjectionHead(
            in_dim=cfg.proj_txt_dim,
            out_dim=cfg.proj_out_dim,
            dropout=cfg.proj_dropout,
        )

    def forward(
        self,
        vision_emb: torch.Tensor,
        text_emb:   torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        vision_emb : FloatTensor  (B, proj_vis_dim)
            Pooled vision representation per sample.
            Typically the CLS token from ContextEncoder or mean-pooled patches.
        text_emb : FloatTensor  (B, proj_txt_dim)
            Pooled [CLS] token from ClinicalBERT for the paired report text.
            Should be detached if you do NOT want gradients to flow into BERT.

        Returns
        -------
        loss : scalar FloatTensor
        info : dict of float diagnostics
            'mean_pos_sim'  — average cosine similarity of positive pairs.
            'mean_neg_sim'  — average cosine similarity of negative pairs.
            'acc_top1'      — % of samples where the correct text is rank-1.
        """
        B = vision_emb.size(0)

        # 1. Project + L2-normalise both modalities
        z_v = self.vision_proj(vision_emb)   # (B, proj_out_dim)
        z_t = self.text_proj(text_emb)       # (B, proj_out_dim)

        # 2. (B × B) cosine similarity matrix, scaled by temperature
        logits = torch.matmul(z_v, z_t.T) / self.temperature  # (B, B)

        # 3. Diagonal = positive pairs (report i ↔ image i)
        labels = torch.arange(B, device=vision_emb.device)

        # 4. Symmetric cross-entropy
        loss_v2t = F.cross_entropy(logits,   labels)   # image  → text direction
        loss_t2v = F.cross_entropy(logits.T, labels)   # text   → image direction
        loss = 0.5 * (loss_v2t + loss_t2v)

        # ── Diagnostics (detached — no graph overhead) ──────────────────────
        with torch.no_grad():
            sim = logits * self.temperature          # unscaled cosine sim
            pos_mask = torch.eye(B, dtype=torch.bool, device=sim.device)
            mean_pos_sim  = sim[pos_mask].mean().item()
            mean_neg_sim  = sim[~pos_mask].mean().item()
            preds         = logits.argmax(dim=1)
            acc_top1      = (preds == labels).float().mean().item()

        info = {
            "mean_pos_sim": mean_pos_sim,
            "mean_neg_sim": mean_neg_sim,
            "acc_top1":     acc_top1,
        }
        return loss, info


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Combined Loss  (the single interface used by train.py)
# ──────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted sum of V-JEPA latent loss and ClinicalBERT distillation loss.

        L_total = λ_jepa · L_jepa  +  λ_distil · L_distil

    Set ``lambda_distil=0.0`` to train purely on the self-supervised JEPA
    objective (useful as a warmup before introducing text alignment).

    Parameters
    ----------
    cfg : LossConfig, optional
        If None, uses default LossConfig values.
    """

    def __init__(self, cfg: Optional[LossConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = LossConfig()
        self.cfg           = cfg
        self.jepa_loss     = VJEPALoss(cfg)
        self.distil_loss   = ClinicalDistilLoss(cfg) if cfg.lambda_distil > 0.0 else None
        self.lambda_jepa   = cfg.lambda_jepa
        self.lambda_distil = cfg.lambda_distil

    def forward(
        self,
        predicted:   torch.Tensor,
        target:      torch.Tensor,
        vision_emb:  Optional[torch.Tensor] = None,
        text_emb:    Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        predicted : FloatTensor  (B, num_masked, embed_dim)
            JEPAPredictor output for masked positions.
        target : FloatTensor  (B, num_masked, embed_dim)
            TargetEncoder output for masked positions (stop-gradient).
        vision_emb : FloatTensor  (B, proj_vis_dim), optional
            Pooled ContextEncoder representation for distillation.
            Required if lambda_distil > 0.
        text_emb : FloatTensor  (B, proj_txt_dim), optional
            ClinicalBERT [CLS] embedding for the paired report.
            Required if lambda_distil > 0.

        Returns
        -------
        total_loss : scalar FloatTensor  — differentiable, call .backward()
        log : dict of float              — logging metrics, no grad attached

        Raises
        ------
        ValueError
            If lambda_distil > 0 but vision_emb or text_emb is None.
        """
        log: Dict[str, float] = {}

        # ── V-JEPA latent loss ──────────────────────────────────────────────
        l_jepa = self.jepa_loss(predicted, target)
        log["loss_jepa"] = l_jepa.item()

        total = self.lambda_jepa * l_jepa

        # ── ClinicalBERT distillation loss ──────────────────────────────────
        if self.lambda_distil > 0.0:
            if vision_emb is None or text_emb is None:
                raise ValueError(
                    "vision_emb and text_emb must be provided when lambda_distil > 0. "
                    "Pass the pooled CLS tokens from ContextEncoder and ClinicalBERT."
                )
            l_distil, distil_info = self.distil_loss(vision_emb, text_emb)
            log["loss_distil"]    = l_distil.item()
            log.update(distil_info)
            total = total + self.lambda_distil * l_distil

        log["loss_total"] = total.item()
        return total, log


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_loss(
    lambda_jepa:    float = 1.0,
    lambda_distil:  float = 0.5,
    temperature:    float = 0.07,
    jepa_loss_type: str   = "mse",
    normalize_jepa: bool  = True,
    proj_vis_dim:   int   = 768,
    proj_txt_dim:   int   = 768,
    proj_out_dim:   int   = 256,
) -> CombinedLoss:
    """Convenience factory for train.py."""
    cfg = LossConfig(
        lambda_jepa=lambda_jepa,
        lambda_distil=lambda_distil,
        temperature=temperature,
        jepa_loss_type=jepa_loss_type,
        normalize_jepa=normalize_jepa,
        proj_vis_dim=proj_vis_dim,
        proj_txt_dim=proj_txt_dim,
        proj_out_dim=proj_out_dim,
    )
    return CombinedLoss(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Smoke Test  (python -m ml_pipeline.loss)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Loss Module Smoke Test")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    B, num_masked, embed_dim = 4, 30, 768
    proj_out = 256

    # ── Synthetic tensors ────────────────────────────────────────────────────
    predicted  = torch.randn(B, num_masked, embed_dim, device=device, requires_grad=True)
    target     = torch.randn(B, num_masked, embed_dim, device=device)  # ← stop-grad in practice

    vision_emb = torch.randn(B, embed_dim, device=device, requires_grad=True)
    text_emb   = torch.randn(B, 768,       device=device)              # ClinicalBERT output

    # ── Combined loss ────────────────────────────────────────────────────────
    criterion = build_loss(
        lambda_jepa=1.0,
        lambda_distil=0.5,
        temperature=0.07,
        jepa_loss_type="mse",
        normalize_jepa=True,
    ).to(device)

    total_loss, log = criterion(predicted, target, vision_emb, text_emb)

    print(f"\n{'─'*40}")
    print(f"  loss_jepa    : {log['loss_jepa']:.6f}")
    print(f"  loss_distil  : {log['loss_distil']:.6f}")
    print(f"  loss_total   : {log['loss_total']:.6f}")
    print(f"  mean_pos_sim : {log['mean_pos_sim']:.4f}")
    print(f"  mean_neg_sim : {log['mean_neg_sim']:.4f}")
    print(f"  acc_top1     : {log['acc_top1']:.4f}")
    print(f"{'─'*40}")

    # ── Backward (MPS-compatible) ────────────────────────────────────────────
    total_loss.backward()
    print(f"\n  predicted grad norm : {predicted.grad.norm().item():.6f}")
    print(f"  vision_emb grad norm: {vision_emb.grad.norm().item():.6f}")

    # ── Smooth-L1 variant ────────────────────────────────────────────────────
    criterion_sl1 = build_loss(jepa_loss_type="smooth_l1", lambda_distil=0.0).to(device)
    l_sl1, log_sl1 = criterion_sl1(predicted.detach(), target)
    print(f"\n  Smooth-L1 JEPA loss (no distil): {log_sl1['loss_jepa']:.6f}")

    print("\n[PASS] All checks passed.")
