"""
ml_pipeline/models/__init__.py
================================
Public API for all ML pipeline model components.

Import from here for clean, stable imports across the codebase:

    from ml_pipeline.models import (
        ContextEncoder, TargetEncoder, EncoderConfig, build_encoder_pair,
        JEPAPredictor, build_predictor,
        VJEPALoss, ClinicalDistilLoss, CombinedLoss, LossConfig, build_loss,
        cosine_ema_schedule,
    )
"""

# ── Encoders (Context + Target ViT) ──────────────────────────────────────────
from ml_pipeline.encoders import (
    EncoderConfig,
    PatchEmbedding,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    VisionTransformer,
    ContextEncoder,
    TargetEncoder,
    build_encoder_pair,
    cosine_ema_schedule,
)

# ── Predictor (V-JEPA lightweight predictor) ─────────────────────────────────
from ml_pipeline.predictor import (
    JEPAPredictor,
    build_predictor,
)

# ── Loss functions ────────────────────────────────────────────────────────────
from ml_pipeline.loss import (
    LossConfig,
    ProjectionHead,
    VJEPALoss,
    ClinicalDistilLoss,
    CombinedLoss,
    build_loss,
)

__all__ = [
    # Encoders
    "EncoderConfig",
    "PatchEmbedding",
    "SinusoidalPositionalEncoding",
    "TransformerBlock",
    "VisionTransformer",
    "ContextEncoder",
    "TargetEncoder",
    "build_encoder_pair",
    "cosine_ema_schedule",
    # Predictor
    "JEPAPredictor",
    "build_predictor",
    # Loss
    "LossConfig",
    "ProjectionHead",
    "VJEPALoss",
    "ClinicalDistilLoss",
    "CombinedLoss",
    "build_loss",
]
