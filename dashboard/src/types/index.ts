// TTT-SSM Phase 0 Dashboard Types

export interface ModelConfig {
  obs_dim: number;
  act_dim: number;
  z_dim: number;
  u_dim: number;
  n_state: number;
  dt: number;
}

export interface PlasticityConfig {
  lr: number;
  weight_decay: number;
  momentum: number;
  nesterov: boolean;
  ns_steps: number;
  adjust_lr_fn: string | null;
  chunk: number;
  buffer_len: number;
  rollback_tol: number;
  grad_norm_max: number;
  state_norm_max: number;
}

export interface SessionMeta {
  schema_version: number;
  session_id: string;
  created_at_unix: number;
  torch_version?: string;
  env_mode: 'linear' | 'nonlinear';
  mu: number;
  base_ckpt_hash: string;
  model_signature: string;
  model_cfg: ModelConfig;
  plasticity_cfg: PlasticityConfig;
}

export interface SessionMetrics {
  baseline_mse_mean: number;
  adaptive_mse_mean: number;
  baseline_mse_last100_mean: number;
  adaptive_mse_last100_mean: number;
  updates_attempted: number;
  updates_committed: number;
  updates_rolled_back: number;
}

export type UpdateStatus =
  | 'commit'
  | 'rollback_loss_regression'
  | 'rollback_grad_norm'
  | 'rollback_state_norm';

export interface UpdateEvent {
  t: number;
  status: UpdateStatus;
  pre_loss: number;
  post_loss: number | null;
  grad_norm: number;
  pre_max_h: number;
  post_max_h?: number;
}

export interface PerStepMetric {
  t: number;
  baseline_mse: number;
  adaptive_mse: number;
  did_update: boolean;
  update_ok: boolean;
}

export interface PlasticWeights {
  W_u: number[][];  // [u_dim, z_dim + act_dim]
  B: number[][];    // [n_state, u_dim]
  W_o: number[][];  // [z_dim, n_state]
}

export interface TrajectoryPoint {
  t: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  ax: number;
  ay: number;
}

export interface SessionData {
  meta: SessionMeta;
  metrics: SessionMetrics;
  perStep: PerStepMetric[];
  updateEvents: UpdateEvent[];
  weights?: PlasticWeights;
  trajectory?: TrajectoryPoint[];
}

export type TabId =
  | 'overview'
  | 'weights'
  | 'transactions'
  | 'environment'
  | 'architecture'
  | 'sessions';

export interface Tab {
  id: TabId;
  label: string;
  icon: string;
}
