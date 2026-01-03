
#!/usr/bin/env python3
"""
Phase 0: Hidden-μ physics + Diagonal Stable SSM + ONLINE WEIGHT UPDATES (TTT) + SESSION PERSISTENCE.

Key properties:
- SSM-only (diagonal stable core)
- Online weight updates during the episode
- Per-session weights + optimizer momentum persisted (resume behaves like a real continuation)
- Baseline run (no updates) vs Adaptive run (updates) on the exact same trajectory
- Rollback semantics (transaction-like): snapshot -> update -> post-check -> commit/rollback
- Encoder is frozen (identity) in Phase 0 to avoid moving-target latent space

This script is self-contained. It will:
1) (Optional) pretrain a base model on random μ sessions
2) Run one session baseline + adaptive
3) Save plots + session artifacts under ./runs/...

Run:
  python3 phase0_muon.py --help
  python3 phase0_muon.py --run_name demo_muon --pretrain_steps 2000 --steps 600 --chunk 32 --env_mode linear
  python3 phase0_muon.py --run_name demo_muon_nl --pretrain_steps 2000 --steps 600 --chunk 32 --env_mode nonlinear
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(obj: object, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def device_auto() -> torch.device:
    # Phase 0 is tiny. CPU is fine and avoids MPS quirks.
    return torch.device("cpu")


def is_2d_param(p: torch.Tensor) -> bool:
    return p.ndim == 2


def newton_schulz_orthogonalize(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """
    Newton–Schulz orthogonalization from Keller Jordan's Muon writeup.
    Returns an approximate "matrix sign" / orthogonalized update direction.

    Reference implementation appears in Keller Jordan's blog post. 
    """
    assert G.ndim == 2, "Muon expects 2D matrices"
    a, b, c = ns_coefficients
    X = G.to(dtype=torch.float32)
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonFallback(torch.optim.Optimizer):
    """
    Minimal Muon optimizer fallback (for environments where torch.optim.Muon doesn't exist).

    - Only supports 2D parameters.
    - Uses SGD-momentum (+ optional Nesterov) then orthogonalizes update via Newton–Schulz.
    - Uses decoupled weight decay (AdamW-style weight decay).

    This is a *toy-friendly* implementation: correctness > speed.

    Muon concept: 
    PyTorch Muon API: 
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: Optional[str] = None,  # "original" | "match_rms_adamw" | None
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if momentum < 0:
            raise ValueError("momentum must be >= 0")
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError("adjust_lr_fn must be None, 'original', or 'match_rms_adamw'")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

        # Validate params now so we fail loudly.
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"MuonFallback only supports 2D params, got shape={tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            mu = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = tuple(group["ns_coefficients"])
            eps = float(group["eps"])
            ns_steps = int(group["ns_steps"])
            adjust_lr_fn = group.get("adjust_lr_fn", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)
                if nesterov:
                    upd = grad.add(buf, alpha=mu)
                else:
                    upd = buf

                O = newton_schulz_orthogonalize(
                    upd,
                    steps=ns_steps,
                    eps=eps,
                    ns_coefficients=ns_coefficients,
                )

                # Decoupled weight decay
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                # Optional LR adjustment for rectangular matrices (roughly matches docs intent).
                lr_eff = lr
                if adjust_lr_fn == "original":
                    m, n = p.shape
                    big = float(max(m, n))
                    small = float(min(m, n))
                    lr_eff = lr * max(1.0, big / small)
                elif adjust_lr_fn == "match_rms_adamw":
                    # The exact Moonshot formula is documented in Muon "is scalable" report.
                    # For this toy fallback, we use a simple stable scaling with sqrt(big).
                    m, n = p.shape
                    big = float(max(m, n))
                    lr_eff = lr * 0.2 * math.sqrt(big)

                p.add_(O, alpha=-lr_eff)

        return loss


def make_muon_optimizer(
    params,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    adjust_lr_fn: Optional[str],
):
    """
    Prefer torch.optim.Muon if available (PyTorch 2.9+), otherwise fallback.
    """
    if hasattr(torch.optim, "Muon"):
        # PyTorch built-in signature: torch.optim.Muon(params, lr=..., weight_decay=..., momentum=..., nesterov=..., ns_steps=..., adjust_lr_fn=...)
        # 
        return torch.optim.Muon(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
    return MuonFallback(
        params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
    )


@dataclass(frozen=True)
class ModelConfig:
    obs_dim: int = 4
    act_dim: int = 2
    z_dim: int = 4  # Phase 0: identity encoder, so z_dim == obs_dim
    u_dim: int = 16
    n_state: int = 32
    dt: float = 1.0


class DiagStableSSM(nn.Module):
    """
    Diagonal stable SSM core:
      h_{t+1} = exp(A*dt) ⊙ h_t + ( [z_t; a_t] W_u^T ) B^T
      ẑ_{t+1} = h_{t+1} W_o^T

    Stability: A = -softplus(a_raw) so A < 0 elementwise.
    In Phase 0 we freeze a_raw (no online updates on stability params).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        d_in = cfg.z_dim + cfg.act_dim

        # 2D matrices we will make plastic (online updates).
        self.W_u = nn.Parameter(torch.randn(cfg.u_dim, d_in) * 0.02)
        self.B = nn.Parameter(torch.randn(cfg.n_state, cfg.u_dim) * 0.02)
        self.W_o = nn.Parameter(torch.randn(cfg.z_dim, cfg.n_state) * 0.02)

        # 1D stability param (frozen in Phase 0)
        a_raw = torch.randn(cfg.n_state) * 0.02
        self.a_raw = nn.Parameter(a_raw)

    def freeze_stability(self) -> None:
        self.a_raw.requires_grad_(False)

    def forward_step(
        self,
        z_t: torch.Tensor,     # [batch, z_dim]
        a_t: torch.Tensor,     # [batch, act_dim]
        h_t: torch.Tensor,     # [batch, n_state]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = -F.softplus(self.a_raw)  # [n_state], negative
        decay = torch.exp(A * self.cfg.dt)  # (0,1)
        decay = decay.unsqueeze(0)  # [1, n_state] for broadcast

        x = torch.cat([z_t, a_t], dim=-1)  # [batch, z_dim+act_dim]
        u = x @ self.W_u.T                 # [batch, u_dim]
        h_next = decay * h_t + (u @ self.B.T)  # [batch, n_state]
        z_pred_next = h_next @ self.W_o.T      # [batch, z_dim]
        return z_pred_next, h_next


@dataclass
class EnvConfig:
    dt: float = 1.0
    noise_std: float = 0.0
    mu_min: float = 0.02
    mu_max: float = 0.25

    # Nonlinear mode: static vs dynamic friction thresholding
    nonlinear: bool = False
    threshold_scale: float = 2.0  # threshold = mu * threshold_scale
    static_mult: float = 2.0      # mu_static = clamp(mu * static_mult, <1)
    dynamic_mult: float = 0.5     # mu_dynamic = mu * dynamic_mult


class HiddenMuPhysicsEnv:
    """
    2D point mass with hidden friction coefficient mu (session-specific).

    State:
      pos ∈ R^2, vel ∈ R^2
    Action:
      accel ∈ R^2

    Linear mode:
      vel <- (1 - mu) * vel + accel
    Nonlinear mode (Phase 0.5):
      if ||vel|| < threshold: use mu_static else mu_dynamic
    """

    def __init__(self, mu: float, cfg: EnvConfig):
        self.mu = float(mu)
        self.cfg = cfg
        self.pos = torch.zeros(2)
        self.vel = torch.zeros(2)

    def reset(self) -> torch.Tensor:
        self.pos.zero_()
        self.vel.zero_()
        return self.observe()

    def observe(self) -> torch.Tensor:
        return torch.cat([self.pos, self.vel], dim=0)  # [4]

    def step(self, accel: torch.Tensor) -> torch.Tensor:
        accel = accel.to(dtype=torch.float32).view(2)

        if not self.cfg.nonlinear:
            mu_eff = self.mu
        else:
            speed = float(torch.linalg.norm(self.vel).item())
            thresh = self.mu * self.cfg.threshold_scale
            mu_static = min(0.95, self.mu * self.cfg.static_mult)
            mu_dynamic = max(0.0, self.mu * self.cfg.dynamic_mult)
            mu_eff = mu_static if speed < thresh else mu_dynamic

        self.vel = (1.0 - mu_eff) * self.vel + accel
        self.pos = self.pos + self.vel * self.cfg.dt

        if self.cfg.noise_std > 0:
            self.pos = self.pos + torch.randn_like(self.pos) * self.cfg.noise_std
            self.vel = self.vel + torch.randn_like(self.vel) * self.cfg.noise_std

        return self.observe()


@dataclass
class PlasticityConfig:
    lr: float = 5e-3
    weight_decay: float = 0.0
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: Optional[str] = None  # None | "original" | "match_rms_adamw"

    chunk: int = 32
    buffer_len: int = 32

    rollback_tol: float = 0.20  # rollback if post_loss > (1+tol)*pre_loss
    grad_norm_max: float = 20.0
    state_norm_max: float = 1e6  # disabled by default (set smaller to guard hidden explosion)


class TransitionBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self.z_t: List[torch.Tensor] = []
        self.a_t: List[torch.Tensor] = []
        self.z_next: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.z_t)

    def append(self, z_t: torch.Tensor, a_t: torch.Tensor, z_next: torch.Tensor) -> None:
        if len(self.z_t) >= self.maxlen:
            self.z_t.pop(0)
            self.a_t.pop(0)
            self.z_next.pop(0)
        self.z_t.append(z_t.detach().cpu())
        self.a_t.append(a_t.detach().cpu())
        self.z_next.append(z_next.detach().cpu())

    def get(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = torch.stack(self.z_t, dim=0).to(device)       # [T, z_dim]
        a_t = torch.stack(self.a_t, dim=0).to(device)       # [T, act_dim]
        z_next = torch.stack(self.z_next, dim=0).to(device) # [T, z_dim]
        return z_t, a_t, z_next


def rollout_loss_on_buffer(
    model: DiagStableSSM,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    z_next: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Compute MSE loss on a transition sequence, by replaying it from h0=0.
    Also returns max hidden state norm, useful for stability guards.
    """
    model.eval()
    with torch.no_grad():
        T = z_t.shape[0]
        h = torch.zeros(1, model.cfg.n_state, device=device)
        preds: List[torch.Tensor] = []
        max_h = 0.0
        for i in range(T):
            z_i = z_t[i].unsqueeze(0)
            a_i = a_t[i].unsqueeze(0)
            z_pred, h = model.forward_step(z_i, a_i, h)
            preds.append(z_pred.squeeze(0))
            hn = float(torch.linalg.norm(h).item())
            if hn > max_h:
                max_h = hn
        pred = torch.stack(preds, dim=0)
        loss = F.mse_loss(pred, z_next)
        return loss, max_h


def grad_global_norm(params: List[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(torch.sum(p.grad.detach() ** 2).item())
    return math.sqrt(total)


def snapshot_params(params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def restore_params(params: List[torch.nn.Parameter], snap: List[torch.Tensor]) -> None:
    assert len(params) == len(snap)
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


def make_model_signature(cfg: ModelConfig, base_ckpt_hash: str) -> str:
    cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    h = hashlib.sha256()
    h.update(cfg_json.encode("utf-8"))
    h.update(base_ckpt_hash.encode("utf-8"))
    return h.hexdigest()


def save_base_checkpoint(path: str, model: DiagStableSSM, cfg: ModelConfig) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({"model_state": model.state_dict(), "cfg": dataclasses.asdict(cfg)}, path)


def load_base_checkpoint(path: str, device: torch.device) -> Tuple[DiagStableSSM, ModelConfig]:
    blob = torch.load(path, map_location=device)
    cfg = ModelConfig(**blob["cfg"])
    model = DiagStableSSM(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.freeze_stability()
    return model, cfg


def save_session(
    session_dir: str,
    schema_version: int,
    model_signature: str,
    base_ckpt_hash: str,
    session_id: str,
    mu: float,
    env_mode: str,
    model: DiagStableSSM,
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, object],
    cfg: ModelConfig,
    p_cfg: PlasticityConfig,
) -> None:
    ensure_dir(session_dir)

    # Save only the plastic weights explicitly.
    plastic_state = {
        "W_u": model.W_u.detach().cpu(),
        "B": model.B.detach().cpu(),
        "W_o": model.W_o.detach().cpu(),
    }
    torch.save(plastic_state, os.path.join(session_dir, "plastic_state.pt"))
    torch.save(optimizer.state_dict(), os.path.join(session_dir, "optim_state.pt"))

    meta = {
        "schema_version": schema_version,
        "session_id": session_id,
        "created_at_unix": int(time.time()),
        "torch_version": torch.__version__,
        "env_mode": env_mode,
        "mu": float(mu),
        "base_ckpt_hash": base_ckpt_hash,
        "model_signature": model_signature,
        "model_cfg": dataclasses.asdict(cfg),
        "plasticity_cfg": dataclasses.asdict(p_cfg),
    }
    json_dump(meta, os.path.join(session_dir, "meta.json"))
    json_dump(metrics, os.path.join(session_dir, "metrics.json"))


def load_session_into_model(
    session_dir: str,
    model: DiagStableSSM,
    optimizer: torch.optim.Optimizer,
    expected_schema_version: int,
    expected_model_signature: str,
    device: torch.device,
) -> Dict[str, object]:
    meta_path = os.path.join(session_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json in session_dir={session_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if int(meta.get("schema_version", -1)) != int(expected_schema_version):
        raise ValueError(f"Schema mismatch: expected {expected_schema_version}, got {meta.get('schema_version')}")

    if str(meta.get("model_signature", "")) != str(expected_model_signature):
        raise ValueError("Model signature mismatch: base model/config changed since session was created")

    plastic_state = torch.load(os.path.join(session_dir, "plastic_state.pt"), map_location=device)
    with torch.no_grad():
        model.W_u.copy_(plastic_state["W_u"].to(device))
        model.B.copy_(plastic_state["B"].to(device))
        model.W_o.copy_(plastic_state["W_o"].to(device))

    optim_state = torch.load(os.path.join(session_dir, "optim_state.pt"), map_location=device)
    optimizer.load_state_dict(optim_state)
    return meta


def pretrain_base_model(
    base_ckpt_path: str,
    cfg: ModelConfig,
    env_cfg: EnvConfig,
    p_cfg: PlasticityConfig,
    steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> None:
    device = device_auto()
    set_seed(seed)

    model = DiagStableSSM(cfg).to(device)
    model.freeze_stability()

    plastic_params = [model.W_u, model.B, model.W_o]
    opt = make_muon_optimizer(
        plastic_params,
        lr=p_cfg.lr,
        weight_decay=p_cfg.weight_decay,
        momentum=p_cfg.momentum,
        nesterov=p_cfg.nesterov,
        ns_steps=p_cfg.ns_steps,
        adjust_lr_fn=p_cfg.adjust_lr_fn,
    )

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        # Build a batch of sequences.
        z0 = torch.zeros(batch_size, cfg.z_dim, device=device)
        a_seq = torch.randn(batch_size, seq_len, cfg.act_dim, device=device) * 0.5

        # Sample mus for each batch element.
        mus = torch.empty(batch_size).uniform_(env_cfg.mu_min, env_cfg.mu_max).tolist()

        # Simulate env in torch (vectorized) for speed.
        pos = torch.zeros(batch_size, 2, device=device)
        vel = torch.zeros(batch_size, 2, device=device)

        # Reset hidden.
        h = torch.zeros(batch_size, cfg.n_state, device=device)

        losses: List[torch.Tensor] = []
        for t in range(seq_len):
            obs = torch.cat([pos, vel], dim=-1)  # [B,4]
            z_t = obs  # identity encoder
            a_t = a_seq[:, t, :]
            z_pred_next, h = model.forward_step(z_t, a_t, h)

            # step env
            mu_t = torch.tensor(mus, device=device).view(batch_size, 1)
            if not env_cfg.nonlinear:
                mu_eff = mu_t
            else:
                speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
                thresh = mu_t * env_cfg.threshold_scale
                mu_static = torch.clamp(mu_t * env_cfg.static_mult, max=0.95)
                mu_dynamic = torch.clamp(mu_t * env_cfg.dynamic_mult, min=0.0)
                mu_eff = torch.where(speed < thresh, mu_static, mu_dynamic)

            vel = (1.0 - mu_eff) * vel + a_t
            pos = pos + vel * env_cfg.dt
            obs_next = torch.cat([pos, vel], dim=-1)
            z_next = obs_next

            losses.append(F.mse_loss(z_pred_next, z_next))

        loss = torch.stack(losses).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1 or step == steps:
            dt = time.time() - t0
            print(f"[pretrain] step={step}/{steps} loss={loss.item():.6f} elapsed_s={dt:.1f}")

    save_base_checkpoint(base_ckpt_path, model, cfg)


def run_one_session(
    run_dir: str,
    base_ckpt_path: str,
    session_id: str,
    mu: float,
    env_cfg: EnvConfig,
    p_cfg: PlasticityConfig,
    steps: int,
    seed: int,
    resume_session_dir: Optional[str],
) -> None:
    import copy
    import csv

    device = device_auto()
    set_seed(seed)

    base_model, cfg = load_base_checkpoint(base_ckpt_path, device=device)

    base_ckpt_hash = sha256_file(base_ckpt_path)
    model_signature = make_model_signature(cfg, base_ckpt_hash)
    schema_version = 1

    # Prepare deterministic action sequence for baseline/adaptive counterfactual.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 12345)
    actions = torch.randn(steps, cfg.act_dim, generator=gen) * 0.5

    # Generate trajectory once (ground truth observations) for counterfactual fairness.
    env = HiddenMuPhysicsEnv(mu=mu, cfg=env_cfg)
    obs0 = env.reset()
    obs = [obs0.clone()]
    for t in range(steps):
        obs.append(env.step(actions[t]))
    obs = torch.stack(obs, dim=0)  # [steps+1, 4]

    # Baseline: no updates
    baseline = DiagStableSSM(cfg).to(device)
    baseline.load_state_dict(base_model.state_dict())
    baseline.freeze_stability()

    h = torch.zeros(1, cfg.n_state, device=device)
    baseline_err: List[float] = []
    for t in range(steps):
        z_t = obs[t].to(device).unsqueeze(0)
        a_t = actions[t].to(device).unsqueeze(0)
        z_true_next = obs[t + 1].to(device).unsqueeze(0)
        z_pred_next, h = baseline.forward_step(z_t, a_t, h)
        baseline_err.append(float(F.mse_loss(z_pred_next, z_true_next).item()))

    # Adaptive: online updates
    model = DiagStableSSM(cfg).to(device)
    model.load_state_dict(base_model.state_dict())
    model.freeze_stability()

    plastic_params = [model.W_u, model.B, model.W_o]
    opt = make_muon_optimizer(
        plastic_params,
        lr=p_cfg.lr,
        weight_decay=p_cfg.weight_decay,
        momentum=p_cfg.momentum,
        nesterov=p_cfg.nesterov,
        ns_steps=p_cfg.ns_steps,
        adjust_lr_fn=p_cfg.adjust_lr_fn,
    )

    if resume_session_dir is not None:
        meta = load_session_into_model(
            resume_session_dir,
            model=model,
            optimizer=opt,
            expected_schema_version=schema_version,
            expected_model_signature=model_signature,
            device=device,
        )
        print(f"[resume] loaded session meta: session_id={meta.get('session_id')} mu={meta.get('mu')} env_mode={meta.get('env_mode')}")

    h = torch.zeros(1, cfg.n_state, device=device)
    adaptive_err: List[float] = []
    update_events: List[Dict[str, object]] = []

    buf = TransitionBuffer(maxlen=p_cfg.buffer_len)

    # CSV logging
    ensure_dir(run_dir)
    csv_path = os.path.join(run_dir, "per_step.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["t", "baseline_mse", "adaptive_mse", "did_update", "update_ok"])
        writer.writeheader()

        for t in range(steps):
            z_t = obs[t].to(device).unsqueeze(0)
            a_t = actions[t].to(device).unsqueeze(0)
            z_true_next = obs[t + 1].to(device).unsqueeze(0)

            z_pred_next, h = model.forward_step(z_t, a_t, h)
            step_mse = float(F.mse_loss(z_pred_next, z_true_next).item())
            adaptive_err.append(step_mse)

            # Update buffer (store CPU tensors)
            buf.append(obs[t], actions[t], obs[t + 1])

            did_update = False
            update_ok = False

            if (t + 1) % p_cfg.chunk == 0 and len(buf) == p_cfg.buffer_len:
                did_update = True

                z_buf, a_buf, z_next_buf = buf.get(device)

                # Pre-check loss on buffer
                pre_loss, pre_max_h = rollout_loss_on_buffer(model, z_buf, a_buf, z_next_buf, device=device)

                # Snapshot weights + optimizer state for rollback.
                snap_w = snapshot_params(plastic_params)
                snap_opt = copy.deepcopy(opt.state_dict())

                # Train on the same buffer (simple + stable)
                model.train()
                opt.zero_grad(set_to_none=True)

                # Replay buffer with grads
                h_train = torch.zeros(1, cfg.n_state, device=device)
                preds: List[torch.Tensor] = []
                for i in range(z_buf.shape[0]):
                    z_i = z_buf[i].unsqueeze(0)
                    a_i = a_buf[i].unsqueeze(0)
                    z_pred, h_train = model.forward_step(z_i, a_i, h_train)
                    preds.append(z_pred.squeeze(0))
                pred = torch.stack(preds, dim=0)
                loss = F.mse_loss(pred, z_next_buf)
                loss.backward()

                gnorm = grad_global_norm(plastic_params)
                if gnorm > p_cfg.grad_norm_max:
                    # Rollback immediately.
                    restore_params(plastic_params, snap_w)
                    opt.load_state_dict(snap_opt)
                    update_ok = False
                    update_events.append(
                        dict(
                            t=t,
                            status="rollback_grad_norm",
                            pre_loss=float(pre_loss.item()),
                            post_loss=None,
                            grad_norm=gnorm,
                            pre_max_h=pre_max_h,
                        )
                    )
                else:
                    opt.step()

                    # Post-check loss on buffer
                    post_loss, post_max_h = rollout_loss_on_buffer(model, z_buf, a_buf, z_next_buf, device=device)

                    rollback = False
                    if float(post_loss.item()) > float(pre_loss.item()) * (1.0 + p_cfg.rollback_tol):
                        rollback = True
                        reason = "rollback_loss_regression"
                    elif post_max_h > p_cfg.state_norm_max:
                        rollback = True
                        reason = "rollback_state_norm"

                    if rollback:
                        restore_params(plastic_params, snap_w)
                        opt.load_state_dict(snap_opt)
                        update_ok = False
                        update_events.append(
                            dict(
                                t=t,
                                status=reason,
                                pre_loss=float(pre_loss.item()),
                                post_loss=float(post_loss.item()),
                                grad_norm=gnorm,
                                pre_max_h=pre_max_h,
                                post_max_h=post_max_h,
                            )
                        )
                    else:
                        update_ok = True
                        update_events.append(
                            dict(
                                t=t,
                                status="commit",
                                pre_loss=float(pre_loss.item()),
                                post_loss=float(post_loss.item()),
                                grad_norm=gnorm,
                                pre_max_h=pre_max_h,
                                post_max_h=post_max_h,
                            )
                        )

                model.eval()

            writer.writerow(
                dict(
                    t=t,
                    baseline_mse=baseline_err[t],
                    adaptive_mse=adaptive_err[t],
                    did_update=int(did_update),
                    update_ok=int(update_ok),
                )
            )

    # Save update events
    json_dump(update_events, os.path.join(run_dir, "update_events.json"))

    # Save session artifacts
    session_dir = os.path.join(run_dir, "session")
    metrics = {
        "baseline_mse_mean": float(sum(baseline_err) / len(baseline_err)),
        "adaptive_mse_mean": float(sum(adaptive_err) / len(adaptive_err)),
        "baseline_mse_last100_mean": float(sum(baseline_err[-100:]) / 100.0),
        "adaptive_mse_last100_mean": float(sum(adaptive_err[-100:]) / 100.0),
        "updates_attempted": int(sum(1 for e in update_events)),
        "updates_committed": int(sum(1 for e in update_events if e.get("status") == "commit")),
        "updates_rolled_back": int(sum(1 for e in update_events if str(e.get("status", "")).startswith("rollback"))),
    }
    save_session(
        session_dir=session_dir,
        schema_version=schema_version,
        model_signature=model_signature,
        base_ckpt_hash=base_ckpt_hash,
        session_id=session_id,
        mu=mu,
        env_mode=("nonlinear" if env_cfg.nonlinear else "linear"),
        model=model,
        optimizer=opt,
        metrics=metrics,
        cfg=cfg,
        p_cfg=p_cfg,
    )

    # Plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(baseline_err, label="baseline (no updates)")
    plt.plot(adaptive_err, label="adaptive (online updates)")
    plt.xlabel("timestep")
    plt.ylabel("MSE(pred z_next, true z_next)")
    plt.title(f"Phase0 HiddenMu | mu={mu:.4f} | env={('nonlinear' if env_cfg.nonlinear else 'linear')}")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "mse_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"[done] run_dir={run_dir}")
    print(f"[metrics] {json.dumps(metrics, indent=2)}")
    print(f"[plot] {plot_path}")
    print(f"[session] {session_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="phase0_muon")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--mu", type=float, default=-1.0, help="If -1, sample uniformly in [mu_min, mu_max].")
    parser.add_argument("--env_mode", type=str, choices=["linear", "nonlinear"], default="linear")

    # Model sizes
    parser.add_argument("--u_dim", type=int, default=16)
    parser.add_argument("--n_state", type=int, default=32)

    # Pretrain
    parser.add_argument("--pretrain_steps", type=int, default=2000)
    parser.add_argument("--pretrain_batch", type=int, default=32)
    parser.add_argument("--pretrain_seq", type=int, default=32)

    # Online plasticity
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--buffer_len", type=int, default=32)
    parser.add_argument("--rollback_tol", type=float, default=0.20)
    parser.add_argument("--grad_norm_max", type=float, default=20.0)
    parser.add_argument("--state_norm_max", type=float, default=1e6)

    # Muon hyperparams
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov", type=int, default=1)
    parser.add_argument("--ns_steps", type=int, default=5)
    parser.add_argument("--adjust_lr_fn", type=str, default="none", choices=["none", "original", "match_rms_adamw"])

    parser.add_argument("--resume_session_dir", type=str, default="", help="Path to previous session dir to resume from.")

    args = parser.parse_args()

    set_seed(args.seed)

    env_cfg = EnvConfig(nonlinear=(args.env_mode == "nonlinear"))
    cfg = ModelConfig(u_dim=args.u_dim, n_state=args.n_state)

    p_cfg = PlasticityConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=bool(args.nesterov),
        ns_steps=args.ns_steps,
        adjust_lr_fn=(None if args.adjust_lr_fn == "none" else args.adjust_lr_fn),
        chunk=args.chunk,
        buffer_len=args.buffer_len,
        rollback_tol=args.rollback_tol,
        grad_norm_max=args.grad_norm_max,
        state_norm_max=args.state_norm_max,
    )

    run_dir = os.path.join(args.run_root, args.run_name)
    ensure_dir(run_dir)

    base_ckpt_path = os.path.join(run_dir, "base_checkpoint.pt")

    if args.pretrain_steps > 0 and not os.path.exists(base_ckpt_path):
        print(f"[info] pretraining base model (steps={args.pretrain_steps}) -> {base_ckpt_path}")
        pretrain_base_model(
            base_ckpt_path=base_ckpt_path,
            cfg=cfg,
            env_cfg=env_cfg,
            p_cfg=p_cfg,
            steps=args.pretrain_steps,
            batch_size=args.pretrain_batch,
            seq_len=args.pretrain_seq,
            seed=args.seed,
        )
    elif not os.path.exists(base_ckpt_path):
        print(f"[info] no pretrain requested; saving random base model -> {base_ckpt_path}")
        model = DiagStableSSM(cfg)
        model.freeze_stability()
        save_base_checkpoint(base_ckpt_path, model, cfg)

    # Choose mu for the session
    if args.mu < 0:
        mu = random.uniform(env_cfg.mu_min, env_cfg.mu_max)
    else:
        mu = float(args.mu)

    resume_dir = args.resume_session_dir.strip() or None

    run_one_session(
        run_dir=run_dir,
        base_ckpt_path=base_ckpt_path,
        session_id="session_001",
        mu=mu,
        env_cfg=env_cfg,
        p_cfg=p_cfg,
        steps=args.steps,
        seed=args.seed,
        resume_session_dir=resume_dir,
    )


if __name__ == "__main__":
    main()
