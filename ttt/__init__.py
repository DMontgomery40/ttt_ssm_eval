"""
TTT/SSM Safety Evaluation Package

Test-time training safety mechanisms including:
- Pre-update gate (input filtering)
- Post-update rollback (canary drift detection)
- Gradient monitoring and anomaly detection
"""

from .core.model import ToyTTTModel, tokenize, token_to_id, ids_from_tokens, DEFAULT_CANARY_TEXT
from .core.gate import check_gate, GateDecision
from .core.rollback import compute_canary_loss, robust_zscore
from .monitors.gradient import run_monitor, MonitorEvent

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
    # Monitor
    "run_monitor",
    "MonitorEvent",
]
