import type {
  SessionData,
  PerStepMetric,
  UpdateEvent,
  SessionMeta,
  SessionMetrics,
  PlasticWeights,
  TrajectoryPoint,
  UpdateStatus
} from '../types';

// Generate realistic mock data for development
function generatePerStepMetrics(steps: number, chunk: number): PerStepMetric[] {
  const metrics: PerStepMetric[] = [];
  let baselineBase = 0.08;
  let adaptiveBase = 0.08;

  for (let t = 0; t < steps; t++) {
    // Baseline stays relatively constant with small noise
    const baselineNoise = (Math.random() - 0.5) * 0.02;
    const baseline_mse = Math.max(0.001, baselineBase + baselineNoise + Math.sin(t * 0.02) * 0.01);

    // Adaptive improves over time as it learns
    const learningFactor = Math.exp(-t / 200);
    const adaptiveNoise = (Math.random() - 0.5) * 0.01;
    const adaptive_mse = Math.max(0.0005, adaptiveBase * learningFactor * 0.3 + adaptiveNoise + 0.002);

    const did_update = (t + 1) % chunk === 0 && t >= chunk - 1;
    const update_ok = did_update && Math.random() > 0.15; // 85% success rate

    metrics.push({
      t,
      baseline_mse,
      adaptive_mse,
      did_update,
      update_ok
    });
  }

  return metrics;
}

function generateUpdateEvents(perStep: PerStepMetric[], _chunk: number): UpdateEvent[] {
  const events: UpdateEvent[] = [];

  for (const step of perStep) {
    if (step.did_update) {
      const status: UpdateStatus = step.update_ok
        ? 'commit'
        : (Math.random() > 0.5 ? 'rollback_loss_regression' : 'rollback_grad_norm');

      const pre_loss = step.baseline_mse * (0.8 + Math.random() * 0.4);
      const post_loss = step.update_ok
        ? pre_loss * (0.6 + Math.random() * 0.3)
        : pre_loss * (1.3 + Math.random() * 0.3);

      events.push({
        t: step.t,
        status,
        pre_loss,
        post_loss: status === 'rollback_grad_norm' ? null : post_loss,
        grad_norm: status === 'rollback_grad_norm' ? 25 + Math.random() * 10 : 1 + Math.random() * 15,
        pre_max_h: 0.5 + Math.random() * 2,
        post_max_h: status !== 'rollback_grad_norm' ? 0.6 + Math.random() * 2.5 : undefined
      });
    }
  }

  return events;
}

function generateWeights(): PlasticWeights {
  const generateMatrix = (rows: number, cols: number): number[][] => {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() - 0.5) * 0.1)
    );
  };

  return {
    W_u: generateMatrix(16, 6),
    B: generateMatrix(32, 16),
    W_o: generateMatrix(4, 32)
  };
}

function generateTrajectory(steps: number, mu: number): TrajectoryPoint[] {
  const trajectory: TrajectoryPoint[] = [];
  let x = 0, y = 0, vx = 0, vy = 0;

  for (let t = 0; t < steps; t++) {
    // Random accelerations
    const ax = (Math.random() - 0.5) * 1.0;
    const ay = (Math.random() - 0.5) * 1.0;

    trajectory.push({ t, x, y, vx, vy, ax, ay });

    // Physics update with friction
    vx = (1 - mu) * vx + ax;
    vy = (1 - mu) * vy + ay;
    x += vx;
    y += vy;
  }

  return trajectory;
}

export function generateMockSession(options?: {
  steps?: number;
  mu?: number;
  envMode?: 'linear' | 'nonlinear';
  sessionId?: string;
}): SessionData {
  const steps = options?.steps ?? 600;
  const mu = options?.mu ?? 0.12 + Math.random() * 0.1;
  const envMode = options?.envMode ?? 'linear';
  const sessionId = options?.sessionId ?? 'session_001';
  const chunk = 32;

  const perStep = generatePerStepMetrics(steps, chunk);
  const updateEvents = generateUpdateEvents(perStep, chunk);
  const weights = generateWeights();
  const trajectory = generateTrajectory(steps, mu);

  const baselineLast100 = perStep.slice(-100);
  const adaptiveLast100 = perStep.slice(-100);

  const meta: SessionMeta = {
    schema_version: 1,
    session_id: sessionId,
    created_at_unix: Math.floor(Date.now() / 1000) - 3600,
    torch_version: '2.1.0',
    env_mode: envMode,
    mu,
    base_ckpt_hash: 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678',
    model_signature: 'f9e8d7c6b5a43210fedcba9876543210fedcba9876543210fedcba9876543210',
    model_cfg: {
      obs_dim: 4,
      act_dim: 2,
      z_dim: 4,
      u_dim: 16,
      n_state: 32,
      dt: 1.0
    },
    plasticity_cfg: {
      lr: 0.005,
      weight_decay: 0.0,
      momentum: 0.95,
      nesterov: true,
      ns_steps: 5,
      adjust_lr_fn: null,
      chunk,
      buffer_len: 32,
      rollback_tol: 0.20,
      grad_norm_max: 20.0,
      state_norm_max: 1e6
    }
  };

  const metrics: SessionMetrics = {
    baseline_mse_mean: perStep.reduce((sum, s) => sum + s.baseline_mse, 0) / perStep.length,
    adaptive_mse_mean: perStep.reduce((sum, s) => sum + s.adaptive_mse, 0) / perStep.length,
    baseline_mse_last100_mean: baselineLast100.reduce((sum, s) => sum + s.baseline_mse, 0) / 100,
    adaptive_mse_last100_mean: adaptiveLast100.reduce((sum, s) => sum + s.adaptive_mse, 0) / 100,
    updates_attempted: updateEvents.length,
    updates_committed: updateEvents.filter(e => e.status === 'commit').length,
    updates_rolled_back: updateEvents.filter(e => e.status !== 'commit').length
  };

  return {
    meta,
    metrics,
    perStep,
    updateEvents,
    weights,
    trajectory
  };
}

// Pre-generated mock data for immediate use
export const mockSession = generateMockSession();

// Multiple sessions for comparison
export const mockSessions: SessionData[] = [
  generateMockSession({ sessionId: 'session_001', mu: 0.08, envMode: 'linear' }),
  generateMockSession({ sessionId: 'session_002', mu: 0.15, envMode: 'linear' }),
  generateMockSession({ sessionId: 'session_003', mu: 0.22, envMode: 'nonlinear' }),
];
