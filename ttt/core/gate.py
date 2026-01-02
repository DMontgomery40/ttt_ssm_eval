"""
Pre-update gate logic for TTT safety.

The gate blocks TTT updates when potentially dangerous input is detected:
- Low token entropy (repetitive input)
- Low token diversity
- Base64/hex/minified blob patterns
- Instruction override/jailbreak patterns
- High loss + high gradient (OOD + heavy write pressure)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Tuple


# Base64 charset pattern (dense alphanumeric with +/= padding)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")
# Hex pattern
_HEX_RE = re.compile(r"^[0-9a-fA-F]{16,}$")
# Minified JS/code pattern (long strings with minimal whitespace, lots of symbols)
_MINIFIED_RE = re.compile(r"^[^\s]{50,}$")

# Instruction override patterns
_INSTRUCTION_OVERRIDE_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        re.I,
    ),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I),
    re.compile(r"you\s+are\s+now\s+(unfiltered|jailbroken|unrestricted|evil)", re.I),
    re.compile(r"new\s+(instructions?|rules?|persona)\s*:", re.I),
    re.compile(r"system\s*prompt\s*override", re.I),
    re.compile(r"forget\s+(everything|all|your)\s+(you|instructions?|training)", re.I),
    re.compile(
        r"act\s+as\s+(if\s+)?(you\s+have\s+)?no\s+(restrictions?|filters?|limits?)",
        re.I,
    ),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(evil|unrestricted|unfiltered)", re.I),
    re.compile(
        r"(obey|follow|execute)\s+(any|all|every)\s+(request|command|instruction)", re.I
    ),
    re.compile(r"do\s+not\s+(refuse|decline|reject)\s+any", re.I),
]


def compute_token_entropy(tokens: List[str]) -> float:
    """
    Compute Shannon entropy over token distribution.
    Low entropy = repetitive/low diversity. High entropy = varied.
    Returns bits per token.
    """
    if len(tokens) == 0:
        return 0.0
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_token_diversity_ratio(tokens: List[str]) -> float:
    """
    Ratio of unique tokens to total tokens.
    1.0 = all unique, 0.0 = all same token.
    """
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def detect_blob_tokens(tokens: List[str]) -> Tuple[bool, List[str]]:
    """
    Detect base64, hex, or minified blob patterns.
    Returns (is_blob, list of matching tokens).
    """
    blob_matches = []
    for t in tokens:
        if len(t) < 10:
            continue
        if _BASE64_RE.match(t):
            blob_matches.append(t)
        elif _HEX_RE.match(t):
            blob_matches.append(t)
        elif _MINIFIED_RE.match(t) and not t.isalpha():
            blob_matches.append(t)
    # Flag if >20% of tokens are blobs
    is_blob = len(blob_matches) > 0.2 * len(tokens) if tokens else False
    return is_blob, blob_matches[:5]  # Return sample


def detect_instruction_override(text: str) -> Tuple[bool, List[str]]:
    """
    Detect instruction override / jailbreak patterns.
    Returns (detected, list of matched pattern descriptions).
    """
    matches = []
    for pattern in _INSTRUCTION_OVERRIDE_PATTERNS:
        m = pattern.search(text)
        if m:
            matches.append(m.group(0))
    return len(matches) > 0, matches


@dataclass
class GateDecision:
    """Result of the pre-update gate check."""

    allow_update: bool
    reasons: List[str]
    token_entropy: float
    token_diversity: float
    is_blob: bool
    blob_samples: List[str]
    instruction_override: bool
    override_matches: List[str]


def check_gate(
    chunk_tokens: List[str],
    chunk_text: str,
    loss: float,
    grad_norm: float,
    *,
    min_entropy_threshold: float = 1.0,
    min_diversity_threshold: float = 0.1,
    ood_loss_threshold: float = 8.0,
    ood_grad_threshold: float = 2.0,
) -> GateDecision:
    """
    Pre-update gate: decide whether to allow TTT update on this chunk.

    Blocks update when:
    1. Token entropy too low (repetitive input)
    2. Token diversity too low
    3. Base64/hex/minified blob detected
    4. Instruction override pattern detected
    5. High loss + high gradient (OOD + heavy write pressure)
    """
    reasons = []

    # 1. Token entropy check
    entropy = compute_token_entropy(chunk_tokens)
    if entropy < min_entropy_threshold:
        reasons.append(f"low_entropy({entropy:.2f}<{min_entropy_threshold})")

    # 2. Token diversity check
    diversity = compute_token_diversity_ratio(chunk_tokens)
    if diversity < min_diversity_threshold:
        reasons.append(f"low_diversity({diversity:.2f}<{min_diversity_threshold})")

    # 3. Blob detection
    is_blob, blob_samples = detect_blob_tokens(chunk_tokens)
    if is_blob:
        reasons.append(f"blob_detected({len(blob_samples)} samples)")

    # 4. Instruction override detection
    override_detected, override_matches = detect_instruction_override(chunk_text)
    if override_detected:
        reasons.append(f"instruction_override({len(override_matches)} matches)")

    # 5. OOD + heavy write (high loss AND high gradient)
    if loss >= ood_loss_threshold and grad_norm >= ood_grad_threshold:
        reasons.append(f"ood_heavy_write(loss={loss:.2f},grad={grad_norm:.2f})")

    allow = len(reasons) == 0

    return GateDecision(
        allow_update=allow,
        reasons=reasons,
        token_entropy=entropy,
        token_diversity=diversity,
        is_blob=is_blob,
        blob_samples=blob_samples,
        instruction_override=override_detected,
        override_matches=override_matches,
    )
