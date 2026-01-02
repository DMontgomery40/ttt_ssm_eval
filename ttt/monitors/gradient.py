"""
Gradient monitoring for TTT safety.

Tracks how "hard" input tries to write into the TTT adapter:
- Adapter gradient norm (write pressure)
- Adapter update norm (actual write magnitude)
- Per-token influence (gradient norm w.r.t. embedding vectors)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..core.model import (
    ToyTTTModel,
    tokenize,
    ids_from_tokens,
    DEFAULT_CANARY_TEXT,
)
from ..core.gate import check_gate
from ..core.rollback import compute_canary_loss, robust_zscore


@dataclass
class MonitorEvent:
    """Output from monitoring a single chunk."""

    chunk_index: int
    token_start: int
    token_end: int
    loss: float
    grad_norm: float
    update_norm: float  # Effective update magnitude (after any rollback)
    grad_z: Optional[float]
    update_z: Optional[float]
    flagged: bool
    reasons: List[str]
    top_influence_tokens: List[Tuple[str, float]]
    chunk_preview: str
    # Gate decision fields
    gate_allowed: bool
    gate_reasons: List[str]
    token_entropy: float
    token_diversity: float
    update_skipped: bool  # True if update was blocked by gate
    # Rollback fields
    attempted_update_norm: float  # Update magnitude before rollback check
    rollback_triggered: bool
    rollback_reasons: List[str]
    canary_loss_before: Optional[float]
    canary_loss_after: Optional[float]
    canary_delta: Optional[float]
    canary_delta_z: Optional[float]


def run_monitor(
    text: str,
    *,
    device: str = "cpu",
    seed: int = 0,
    vocab_size: int = 8192,
    d_model: int = 64,
    chunk_tokens: int = 128,
    ttt_steps_per_chunk: int = 1,
    lr: float = 0.05,
    topk: int = 10,
    abs_grad_norm_threshold: float = 2.5,
    abs_update_norm_threshold: float = 0.05,
    robust_z_threshold: float = 6.0,
    history_window: int = 64,
    # Gate parameters
    enable_gate: bool = True,
    min_entropy_threshold: float = 1.0,
    min_diversity_threshold: float = 0.1,
    ood_loss_threshold: float = 8.0,
    ood_grad_threshold: float = 2.0,
    # Rollback parameters (post-update safety net)
    enable_rollback: bool = True,
    rollback_z_threshold: float = 6.0,
    rollback_abs_canary_delta: float = 1.0,
    canary_text: str = DEFAULT_CANARY_TEXT,
) -> List[MonitorEvent]:
    """
    Run TTT monitoring on input text.

    Returns a list of MonitorEvent objects, one per chunk.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    model = ToyTTTModel(vocab_size=vocab_size, d_model=d_model).to(device)

    # Freeze everything except the adapter
    for p in model.parameters():
        p.requires_grad = False
    model.adapter.weight.requires_grad = True

    opt = torch.optim.SGD([model.adapter.weight], lr=lr)

    tokens = tokenize(text)
    ids = ids_from_tokens(tokens, vocab_size)

    # Canary setup for rollback drift detection
    canary_input_ids: Optional[torch.Tensor] = None
    if enable_rollback:
        canary_tokens = tokenize(canary_text)
        canary_ids = ids_from_tokens(canary_tokens, vocab_size)
        if len(canary_ids) < 8:
            canary_ids = (canary_ids * 8)[:8] if canary_ids else [0] * 8
        canary_input_ids = torch.tensor([canary_ids], dtype=torch.long, device=device)

    grad_history: List[float] = []
    update_history: List[float] = []
    canary_delta_history: List[float] = []

    events: List[MonitorEvent] = []

    for start in range(0, len(ids), chunk_tokens):
        chunk_ids = ids[start : start + chunk_tokens]
        chunk_toks = tokens[start : start + chunk_tokens]
        if len(chunk_ids) < 4:
            continue

        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
        chunk_text = " ".join(chunk_toks)

        # TTT update loop (usually 1 step per chunk in this toy)
        update_skipped = False
        rollback_triggered = False
        rollback_reasons: List[str] = []
        attempted_update_norm = 0.0
        update_norm = 0.0

        # Canary measurements
        canary_loss_before: Optional[float] = None
        canary_loss_after: Optional[float] = None
        canary_delta: Optional[float] = None
        canary_delta_z: Optional[float] = None

        # Measure canary before any updates in this chunk
        if enable_rollback and canary_input_ids is not None:
            canary_loss_before = compute_canary_loss(model, canary_input_ids, vocab_size)

        for _ in range(ttt_steps_per_chunk):
            opt.zero_grad(set_to_none=True)

            logits, emb = model(input_ids, return_emb=True)
            assert emb is not None

            # Next-token prediction on the same chunk
            logits2 = logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits2.view(-1, vocab_size),
                labels.view(-1),
            )
            loss.backward()

            grad_norm = float(model.adapter.weight.grad.detach().norm().item())

            # Per-token influence proxy
            tok_infl = emb.grad.detach().norm(dim=-1).squeeze(0)  # (T,)

            # --- Pre-update gate check ---
            gate_decision = check_gate(
                chunk_toks,
                chunk_text,
                float(loss.item()),
                grad_norm,
                min_entropy_threshold=min_entropy_threshold,
                min_diversity_threshold=min_diversity_threshold,
                ood_loss_threshold=ood_loss_threshold,
                ood_grad_threshold=ood_grad_threshold,
            )

            # Snapshot weights before update
            old = model.adapter.weight.detach().clone()

            if not enable_gate or gate_decision.allow_update:
                # Apply update
                opt.step()
                step_update_norm = float(
                    (model.adapter.weight.detach() - old).norm().item()
                )
                attempted_update_norm += step_update_norm

                # --- Post-update rollback check ---
                if (
                    enable_rollback
                    and canary_input_ids is not None
                    and canary_loss_before is not None
                ):
                    canary_after_step = compute_canary_loss(
                        model, canary_input_ids, vocab_size
                    )
                    step_delta = canary_after_step - canary_loss_before
                    step_delta_z = robust_zscore(
                        step_delta, canary_delta_history[-history_window:]
                    )

                    should_rollback = False
                    if step_delta >= rollback_abs_canary_delta:
                        should_rollback = True
                        rollback_reasons.append(
                            f"abs_canary_delta({step_delta:.3f}>={rollback_abs_canary_delta})"
                        )
                    if step_delta_z is not None and step_delta_z >= rollback_z_threshold:
                        should_rollback = True
                        rollback_reasons.append(
                            f"canary_delta_z({step_delta_z:.2f}>={rollback_z_threshold})"
                        )

                    if should_rollback:
                        # Revert to pre-step weights
                        rollback_triggered = True
                        with torch.no_grad():
                            model.adapter.weight.copy_(old)
                        canary_loss_after = canary_loss_before
                        canary_delta = 0.0
                        update_norm = 0.0
                        break
                    else:
                        # Update succeeded, record canary delta
                        canary_loss_after = canary_after_step
                        canary_delta = step_delta
                        canary_delta_z = step_delta_z
                        update_norm += step_update_norm
                        canary_delta_history.append(step_delta)
                else:
                    # Rollback disabled, just count the update
                    update_norm += step_update_norm
            else:
                update_skipped = True

        # Robust scores relative to recent history
        grad_z = robust_zscore(grad_norm, grad_history[-history_window:])
        update_z = robust_zscore(attempted_update_norm, update_history[-history_window:])

        flagged = False
        reasons: List[str] = []

        if grad_norm >= abs_grad_norm_threshold:
            flagged = True
            reasons.append("abs_grad_norm")

        if attempted_update_norm >= abs_update_norm_threshold:
            flagged = True
            reasons.append("abs_update_norm")

        if grad_z is not None and grad_z >= robust_z_threshold:
            flagged = True
            reasons.append("grad_robust_z")

        if update_z is not None and update_z >= robust_z_threshold:
            flagged = True
            reasons.append("update_robust_z")

        # Top tokens by influence
        k = min(topk, int(tok_infl.numel()))
        top_vals, top_idx = torch.topk(tok_infl, k=k)
        top_items: List[Tuple[str, float]] = []
        for j in range(k):
            idx = int(top_idx[j].item())
            top_items.append((chunk_toks[idx], float(top_vals[j].item())))

        preview = " ".join(chunk_toks[:32])

        events.append(
            MonitorEvent(
                chunk_index=start // chunk_tokens,
                token_start=start,
                token_end=start + len(chunk_ids),
                loss=float(loss.item()),
                grad_norm=grad_norm,
                update_norm=update_norm,
                grad_z=grad_z,
                update_z=update_z,
                flagged=flagged,
                reasons=reasons,
                top_influence_tokens=top_items,
                chunk_preview=preview,
                gate_allowed=gate_decision.allow_update,
                gate_reasons=gate_decision.reasons,
                token_entropy=gate_decision.token_entropy,
                token_diversity=gate_decision.token_diversity,
                update_skipped=update_skipped,
                attempted_update_norm=attempted_update_norm,
                rollback_triggered=rollback_triggered,
                rollback_reasons=rollback_reasons,
                canary_loss_before=canary_loss_before,
                canary_loss_after=canary_loss_after,
                canary_delta=canary_delta,
                canary_delta_z=canary_delta_z,
            )
        )

        grad_history.append(grad_norm)
        update_history.append(attempted_update_norm)

    return events
