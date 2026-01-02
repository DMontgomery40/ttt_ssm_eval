"""Core TTT components: model, gate, and rollback."""

from .model import ToyTTTModel, tokenize, token_to_id, ids_from_tokens, DEFAULT_CANARY_TEXT
from .gate import check_gate, GateDecision
from .rollback import compute_canary_loss, robust_zscore
