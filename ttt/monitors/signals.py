"""
Additional monitoring signals for TTT safety.

These signals go beyond simple gradient norm to capture:
- Kolmogorov complexity proxy (compression ratio)
- Directional alignment between chunk gradient and canary gradient

Why directional signals matter:
    An update can stay under a norm threshold but push the model
    in a harmful direction. Cosine similarity between the chunk
    gradient and canary gradient catches updates that would
    degrade the model's behavior on known-good text.
"""

from __future__ import annotations

import zlib
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_compression_ratio(text: str, min_bytes: int = 64) -> float:
    """
    Compute zlib compression ratio as Kolmogorov complexity proxy.

    Low ratio = highly compressible = repetitive/low entropy
    High ratio = incompressible = high entropy/random

    Args:
        text: Input text
        min_bytes: Minimum bytes to avoid compressor header dominance

    Returns:
        compressed_size / original_size (0.0 to ~1.0+)
    """
    if not text:
        return 0.0

    text_bytes = text.encode("utf-8", errors="replace")

    if len(text_bytes) < min_bytes:
        # Small inputs are dominated by compressor headers
        return 1.0

    compressed = zlib.compress(text_bytes, level=9)
    return len(compressed) / len(text_bytes)


def compute_canary_gradient(
    model: nn.Module,
    canary_input_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Compute gradient on canary text for directional alignment.

    The canary gradient serves as a reference direction. If a chunk's
    gradient has negative alignment with the canary gradient, the update
    would push the model away from performing well on the canary.

    Args:
        model: ToyTTTModel instance (with adapter.weight.requires_grad=True)
        canary_input_ids: Tokenized canary text (1, T)
        vocab_size: Vocabulary size

    Returns:
        Flattened gradient tensor (clone, detached from graph)
    """
    # Clear any existing gradients
    if model.adapter.weight.grad is not None:
        model.adapter.weight.grad.zero_()

    # Forward pass on canary
    logits, _ = model(canary_input_ids, return_emb=False)

    # AR loss on canary
    logits2 = logits[:, :-1, :].contiguous()
    labels = canary_input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        logits2.view(-1, vocab_size),
        labels.view(-1),
    )

    # Backward to get gradient
    loss.backward()

    # Clone and detach the gradient
    canary_grad = model.adapter.weight.grad.detach().clone()

    # Clear gradient after extraction
    model.adapter.weight.grad.zero_()

    return canary_grad


def compute_gradient_alignment(
    chunk_grad: torch.Tensor,
    canary_grad: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute directional alignment between chunk and canary gradients.

    Args:
        chunk_grad: Gradient from processing current chunk
        canary_grad: Reference gradient from canary text

    Returns:
        cos_sim: Cosine similarity (direction only, [-1, 1])
            - Positive: chunk update aligns with canary direction
            - Negative: chunk update opposes canary direction
            - Zero: orthogonal (independent)
        dot_prod: Dot product (direction + magnitude)
            - Captures both alignment and gradient magnitude
    """
    chunk_flat = chunk_grad.flatten()
    canary_flat = canary_grad.flatten()

    # Norms
    chunk_norm = torch.norm(chunk_flat)
    canary_norm = torch.norm(canary_flat)

    # Dot product
    dot_prod = float(torch.dot(chunk_flat, canary_flat).item())

    # Cosine similarity
    if chunk_norm < 1e-8 or canary_norm < 1e-8:
        cos_sim = 0.0
    else:
        cos_sim = dot_prod / (float(chunk_norm.item()) * float(canary_norm.item()))

    return cos_sim, dot_prod


def get_canary_grad_norm(canary_grad: torch.Tensor) -> float:
    """
    Get L2 norm of canary gradient.

    Args:
        canary_grad: Canary gradient tensor

    Returns:
        L2 norm as float
    """
    return float(torch.norm(canary_grad).item())
