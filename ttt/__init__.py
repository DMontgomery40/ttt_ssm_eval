"""
TTT/SSM Safety Evaluation Package

Test-time training safety mechanisms including:
- Pre-update gate (input filtering)
- Post-update rollback (canary drift detection)
- Gradient monitoring and anomaly detection
- Multiple backbone architectures (GRU, SSM)
- Multiple TTT objectives (AR, MLM)
"""

from .core.model import ToyTTTModel, tokenize, token_to_id, ids_from_tokens, DEFAULT_CANARY_TEXT
from .core.gate import check_gate, GateDecision
from .core.rollback import compute_canary_loss, robust_zscore
from .core.backbone import (
    BackboneType,
    BaseBackbone,
    GRUBackbone,
    DiagonalSelectiveSSM,
    create_backbone,
)
from .core.objective import (
    ObjectiveType,
    compute_ar_loss,
    compute_mlm_loss,
    create_mlm_mask,
    compute_objective_loss,
)
from .monitors.gradient import run_monitor, MonitorEvent
from .monitors.signals import (
    compute_compression_ratio,
    compute_canary_gradient,
    compute_gradient_alignment,
    get_canary_grad_norm,
)

__all__ = [
    # Model
    "ToyTTTModel",
    "tokenize",
    "token_to_id",
    "ids_from_tokens",
    "DEFAULT_CANARY_TEXT",
    # Gate
    "check_gate",
    "GateDecision",
    # Rollback
    "compute_canary_loss",
    "robust_zscore",
    # Backbone
    "BackboneType",
    "BaseBackbone",
    "GRUBackbone",
    "DiagonalSelectiveSSM",
    "create_backbone",
    # Objective
    "ObjectiveType",
    "compute_ar_loss",
    "compute_mlm_loss",
    "create_mlm_mask",
    "compute_objective_loss",
    # Monitor
    "run_monitor",
    "MonitorEvent",
    # Signals
    "compute_compression_ratio",
    "compute_canary_gradient",
    "compute_gradient_alignment",
    "get_canary_grad_norm",
]
