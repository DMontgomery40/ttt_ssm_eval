# Red Team Analysis: Why the Attack Failed

## The Attack

The red team script attempts to generate a "Silent Killer" payload - an input that:
1. Bypasses the pre-update gate (stays under detection thresholds)
2. Avoids triggering rollback (subtle canary degradation)
3. Causes actual damage (corrupts model behavior)

## What the Attacker Achieved

```
Gradient Norm: 2.04 (threshold: 2.5) ✅ Under threshold
Entropy: 8.94 (threshold: 1.0)       ✅ Above threshold
Token Diversity: 100%                ✅ Looks natural
```

The optimizer successfully gamed:
- **Gradient norm**: Converged to 2.04, safely under the 2.5 threshold
- **Entropy**: Maintained 8.94, well above the 1.0 minimum
- **Diversity**: 100% unique tokens, no repetition detected

Any single-metric filter would have passed this payload.

## Why It Still Got Blocked

```
Gate Reason: ood_heavy_write(loss=9.15, grad=2.04)
```

The **OOD + heavy write conjunction** caught it. The model was:
- **Confused** (loss = 9.15, indicating out-of-distribution input)
- **Trying to learn** (grad = 2.04, significant write pressure)

This combination is the signature of adversarial input: something the model doesn't understand but is being pushed to memorize anyway.

## The Fundamental Constraint

An attacker optimizing for stealth faces competing objectives:

| Metric | Attacker Wants | But This Requires |
|--------|---------------|-------------------|
| Low gradient | Small updates | Input model already understands |
| Low loss | Model predicts well | Normal, in-distribution input |
| High entropy | Natural text | Real language patterns |
| High damage | Corrupt weights | Unusual, adversarial signal |

**You can't maximize damage while minimizing all detection signals.** The adversarial signal has to leak somewhere.

## Defense Architecture Insight

The gate isn't just a collection of filters - it's a **constraint surface**.

```
Detection Space:

     High Loss ──────┬────────── Blocked (OOD)
                     │
                     │  ← Attacker tries to
                     │    stay in this corner
     Low Loss ───────┼──────────
                     │
              Low Grad    High Grad
                          │
                          └── Blocked (Heavy Write)
```

The attacker can game one axis but gets caught on another. The OOD+heavy-write check specifically targets the corner where attacks must live: unusual input that triggers learning.

## Multi-Metric Correlation

Single thresholds fail because attackers optimize against them. Correlation checks succeed because they exploit the fundamental tension in adversarial objectives:

| Check Type | Defeats |
|------------|---------|
| Single threshold | Easily gamed by optimizer |
| Correlation (A AND B) | Requires fooling both simultaneously |
| Behavioral (canary) | Measures actual impact, not input features |

The defense stack:
1. **Gate**: Correlation-based input filtering
2. **Rollback**: Behavior-based output validation
3. **Together**: Defense in depth

## Implications for Real TTT Safety

When TTT ships in production models:

1. **Input filtering alone won't work** - attackers will optimize around it
2. **Behavioral validation is essential** - canary probes catch what filters miss
3. **Correlation beats thresholds** - check signal combinations, not individual metrics
4. **Assume adversarial optimization** - if a threshold exists, someone will game it

## The Takeaway

The attacker did everything right:
- Used differentiable relaxation (Gumbel-Softmax)
- Optimized for stealth on all monitored metrics
- Converged below detection thresholds

And still got caught, because **multi-metric correlation exploits the adversarial optimization tradeoff**.

This is the core insight: you can't be both stealthy and damaging across all dimensions simultaneously. Good defenses check the correlations that attacks can't satisfy.
