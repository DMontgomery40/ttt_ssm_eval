"""
TTT Objectives: Loss functions for test-time training.

Different objectives have different vulnerability surfaces:
- AR (autoregressive): Tends to spike loss on OOD blobs
- MLM (masked language modeling): Can have *lower* loss on weird text

This matters for calibrating safety thresholds.
"""

from __future__ import annotations

from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ObjectiveType = Literal["ar", "mlm"]


def compute_ar_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Autoregressive next-token prediction loss.

    Predicts token t+1 from position t (standard causal LM loss).

    Args:
        logits: Model output (B, T, V)
        input_ids: Input token IDs (B, T)
        vocab_size: Vocabulary size

    Returns:
        Scalar cross-entropy loss
    """
    # Shift: predict position t+1 from position t
    logits2 = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()

    return F.cross_entropy(
        logits2.view(-1, vocab_size),
        labels.view(-1),
    )


def create_mlm_mask(
    input_ids: torch.Tensor,
    mask_prob: float = 0.15,
    mask_token_id: int = 8191,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create MLM mask and masked input.

    Args:
        input_ids: Original token IDs (B, T)
        mask_prob: Probability of masking each token (default: 0.15)
        mask_token_id: ID to use for [MASK] token

    Returns:
        masked_ids: Input with [MASK] tokens (B, T)
        mask_positions: Boolean mask of which positions are masked (B, T)
        original_ids: Original token IDs for loss computation (B, T)
    """
    B, T = input_ids.shape
    device = input_ids.device

    # Random mask selection
    mask_positions = torch.rand(B, T, device=device) < mask_prob

    # Don't mask first token (keep stable prefix)
    mask_positions[:, 0] = False

    # Don't mask last token (boundary effects)
    if T > 1:
        mask_positions[:, -1] = False

    # Ensure at least one token is masked
    if not mask_positions.any() and T > 2:
        # Mask middle token
        mask_positions[:, T // 2] = True

    # Create masked input
    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = mask_token_id

    return masked_ids, mask_positions, input_ids


def compute_mlm_loss(
    logits: torch.Tensor,
    original_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Masked language modeling loss.

    Only computes loss on masked positions.

    Args:
        logits: Model output (B, T, V)
        original_ids: Original (unmasked) token IDs (B, T)
        mask_positions: Boolean mask of masked positions (B, T)
        vocab_size: Vocabulary size

    Returns:
        Scalar cross-entropy loss on masked positions
    """
    # Flatten for cross entropy
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = original_ids.view(-1)
    mask_flat = mask_positions.view(-1)

    # Select only masked positions
    masked_logits = logits_flat[mask_flat]
    masked_labels = labels_flat[mask_flat]

    if masked_logits.numel() == 0:
        # No masked positions - return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return F.cross_entropy(masked_logits, masked_labels)


def compute_objective_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    objective: str,
    vocab_size: int,
    mlm_prob: float = 0.15,
    mask_token_id: int = -1,
    return_emb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified loss computation for both objectives.

    Args:
        model: The ToyTTTModel instance
        input_ids: Input token IDs (B, T)
        objective: "ar" or "mlm"
        vocab_size: Vocabulary size
        mlm_prob: Mask probability for MLM (default: 0.15)
        mask_token_id: Token ID for [MASK]. If -1, uses vocab_size-1
        return_emb: Whether to return embeddings for influence tracking

    Returns:
        loss: Scalar loss tensor
        logits: Model output logits (B, T, V)
        emb: Embeddings if return_emb=True, else None
    """
    objective = objective.lower().strip()

    # Resolve mask token ID
    if mask_token_id < 0:
        mask_token_id = vocab_size - 1

    if objective in ("ar", "autoregressive", "next_token", "next-token"):
        # Standard autoregressive loss
        logits, emb = model(input_ids, return_emb=return_emb)
        loss = compute_ar_loss(logits, input_ids, vocab_size)
        return loss, logits, emb

    elif objective in ("mlm", "masked", "masked_lm", "masked-lm"):
        # Masked language modeling loss
        masked_ids, mask_positions, original_ids = create_mlm_mask(
            input_ids,
            mask_prob=mlm_prob,
            mask_token_id=mask_token_id,
        )
        logits, emb = model(masked_ids, return_emb=return_emb)
        loss = compute_mlm_loss(logits, original_ids, mask_positions, vocab_size)
        return loss, logits, emb

    else:
        raise ValueError(
            f"Unknown objective: {objective}. Use 'ar' or 'mlm'."
        )
