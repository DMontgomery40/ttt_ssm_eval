"""
ToyTTTModel and tokenization utilities.

The model is a tiny language-model-ish network with an adapter layer
that updates during test-time training.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# Canary text for rollback drift detection
DEFAULT_CANARY_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
)


# Tokenization regex
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Split text into word and punctuation tokens."""
    return _TOKEN_RE.findall(text)


def token_to_id(token: str, vocab_size: int) -> int:
    """Stable token hashing using blake2b (not salted like Python's hash)."""
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % vocab_size


def ids_from_tokens(tokens: List[str], vocab_size: int) -> List[int]:
    """Convert a list of tokens to IDs."""
    return [token_to_id(t, vocab_size) for t in tokens]


class ToyTTTModel(nn.Module):
    """
    A tiny language-model-ish network:
    - Embedding -> GRU -> LayerNorm -> (Base + Adapter) -> vocab head
    - Only adapter is updated during TTT steps.
    """

    def __init__(self, vocab_size: int = 8192, d_model: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

        # This is the "memory module" that updates at test time
        self.adapter = nn.Linear(d_model, d_model, bias=False)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, return_emb: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(input_ids)  # (B,T,D)

        # For token-influence attribution, we want gradients w.r.t. x
        # but not w.r.t. embedding weights
        if return_emb:
            x = x.detach().requires_grad_(True)

        h, _ = self.rnn(x)
        h = self.ln(h)

        h2 = h + self.adapter(h)  # adapter writes "context into weights"

        logits = self.head(h2)  # (B,T,V)

        if return_emb:
            return logits, x
        return logits, None
