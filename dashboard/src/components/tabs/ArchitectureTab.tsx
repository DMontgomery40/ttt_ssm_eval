import { useMemo } from 'react';
import { useDashboardStore } from '../../store';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export function ArchitectureTab() {
  const { currentSession } = useDashboardStore();
  const { meta } = currentSession;
  const { model_cfg, plasticity_cfg } = meta;

  // Parameter breakdown
  const paramBreakdown = useMemo(() => [
    {
      name: 'W_u',
      shape: `[${model_cfg.u_dim}, ${model_cfg.z_dim + model_cfg.act_dim}]`,
      count: model_cfg.u_dim * (model_cfg.z_dim + model_cfg.act_dim),
      plastic: true,
      color: '#58a6ff',
    },
    {
      name: 'B',
      shape: `[${model_cfg.n_state}, ${model_cfg.u_dim}]`,
      count: model_cfg.n_state * model_cfg.u_dim,
      plastic: true,
      color: '#39d353',
    },
    {
      name: 'W_o',
      shape: `[${model_cfg.z_dim}, ${model_cfg.n_state}]`,
      count: model_cfg.z_dim * model_cfg.n_state,
      plastic: true,
      color: '#a371f7',
    },
    {
      name: 'a_raw',
      shape: `[${model_cfg.n_state}]`,
      count: model_cfg.n_state,
      plastic: false,
      color: '#6e7681',
    },
  ], [model_cfg]);

  const totalPlastic = paramBreakdown.filter(p => p.plastic).reduce((a, b) => a + b.count, 0);
  const totalFrozen = paramBreakdown.filter(p => !p.plastic).reduce((a, b) => a + b.count, 0);

  return (
    <div className="space-y-6">
      {/* Architecture Diagram */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-text-primary mb-6">Diagonal Stable SSM Architecture</h3>

        {/* Flow diagram */}
        <div className="flex items-center justify-center gap-2 overflow-x-auto py-4">
          {/* Input */}
          <DiagramBox label="[z_t, a_t]" sublabel={`[${model_cfg.z_dim + model_cfg.act_dim}]`} color="#8b949e" />
          <Arrow />

          {/* W_u */}
          <DiagramBox label="W_u" sublabel="Input proj" color="#58a6ff" plastic />
          <Arrow />

          {/* u_t */}
          <DiagramBox label="u_t" sublabel={`[${model_cfg.u_dim}]`} color="#8b949e" />
          <Arrow />

          {/* B */}
          <DiagramBox label="B" sublabel="State update" color="#39d353" plastic />
          <Arrow />

          {/* SSM core */}
          <div className="bg-surface-100 border-2 border-accent-orange rounded-lg p-4 text-center">
            <div className="font-mono text-sm text-accent-orange mb-1">SSM Core</div>
            <div className="text-xs text-text-muted">decay ⊙ h_t + Bu</div>
            <div className="text-xs text-text-muted mt-1">decay = exp(A·dt)</div>
          </div>
          <Arrow />

          {/* h_t+1 */}
          <DiagramBox label="h_{t+1}" sublabel={`[${model_cfg.n_state}]`} color="#8b949e" />
          <Arrow />

          {/* W_o */}
          <DiagramBox label="W_o" sublabel="Output proj" color="#a371f7" plastic />
          <Arrow />

          {/* Output */}
          <DiagramBox label="ẑ_{t+1}" sublabel={`[${model_cfg.z_dim}]`} color="#39d353" highlight />
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-6 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded border-2 border-dashed border-accent-green" />
            <span className="text-text-muted">Plastic (updated at test time)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-surface-100" />
            <span className="text-text-muted">Frozen</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded border-2 border-accent-orange" />
            <span className="text-text-muted">Stability-guaranteed core</span>
          </div>
        </div>
      </div>

      {/* Stability Explanation */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Stability Guarantee</h3>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-text-secondary mb-4">
              The SSM uses a diagonal state transition with guaranteed stability.
              The decay parameters are constrained to be in (0, 1), preventing
              hidden state explosion.
            </p>

            <div className="bg-surface-100 rounded-lg p-4 font-mono text-sm space-y-2">
              <div>
                <span className="text-text-muted">A</span>
                <span className="text-text-primary mx-2">=</span>
                <span className="text-accent-orange">-softplus(a_raw)</span>
                <span className="text-text-muted ml-2">{'// A < 0 always'}</span>
              </div>
              <div>
                <span className="text-text-muted">decay</span>
                <span className="text-text-primary mx-2">=</span>
                <span className="text-accent-green">exp(A × dt)</span>
                <span className="text-text-muted ml-2">{'// decay ∈ (0, 1)'}</span>
              </div>
            </div>
          </div>

          <div className="bg-accent-green/10 border border-accent-green/30 rounded-lg p-4">
            <p className="text-accent-green font-medium mb-2">Why this matters</p>
            <p className="text-sm text-text-secondary">
              With |decay| {'<'} 1 for all hidden dimensions, the hidden state
              naturally decays toward zero. This means even if online updates
              introduce errors, the system self-corrects and cannot explode.
              The model is <strong>stable by construction</strong>.
            </p>
          </div>
        </div>
      </div>

      {/* Parameter Count Chart */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Parameter Distribution</h3>

        <div className="grid grid-cols-2 gap-6">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={paramBreakdown} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis type="number" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => [value.toLocaleString(), 'Parameters']}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {paramBreakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="space-y-4">
            {paramBreakdown.map((param) => (
              <div key={param.name} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: param.color }} />
                  <span className="font-mono">{param.name}</span>
                  <span className="text-xs text-text-muted">{param.shape}</span>
                </div>
                <div className="text-right">
                  <span className="font-mono font-bold">{param.count.toLocaleString()}</span>
                  <span
                    className={`ml-2 text-xs px-1.5 py-0.5 rounded ${
                      param.plastic ? 'bg-accent-green/20 text-accent-green' : 'bg-surface-200 text-text-muted'
                    }`}
                  >
                    {param.plastic ? 'plastic' : 'frozen'}
                  </span>
                </div>
              </div>
            ))}

            <div className="pt-4 border-t border-surface-200">
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Total plastic:</span>
                <span className="font-mono font-bold text-accent-green">{totalPlastic.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm mt-1">
                <span className="text-text-muted">Total frozen:</span>
                <span className="font-mono font-bold text-text-secondary">{totalFrozen.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Muon Optimizer Explainer */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-6">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Muon Optimizer</h3>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-text-secondary mb-4">
              Unlike AdamW, Muon orthogonalizes gradients using Newton-Schulz iteration
              before applying updates. This produces more stable training dynamics,
              especially important for test-time learning where we can't afford
              divergence.
            </p>

            <div className="bg-surface-100 rounded-lg p-4 space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-text-muted">Learning rate</span>
                <span className="font-mono text-accent-blue">{plasticity_cfg.lr}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Momentum</span>
                <span className="font-mono text-accent-blue">{plasticity_cfg.momentum}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Nesterov</span>
                <span className="font-mono text-accent-blue">{plasticity_cfg.nesterov ? 'Yes' : 'No'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">NS iterations</span>
                <span className="font-mono text-accent-blue">{plasticity_cfg.ns_steps}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Weight decay</span>
                <span className="font-mono text-accent-blue">{plasticity_cfg.weight_decay}</span>
              </div>
            </div>
          </div>

          <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-4">
            <p className="text-accent-blue font-medium mb-2">Newton-Schulz Orthogonalization</p>
            <p className="text-sm text-text-secondary mb-3">
              Each gradient matrix is iteratively transformed to approximate
              an orthogonal matrix, preventing directional collapse.
            </p>
            <code className="block bg-surface rounded p-2 font-mono text-xs text-text-primary">
              X ← a·X + b·(X·X^T)·X + c·(X·X^T)^2·X
            </code>
            <p className="text-xs text-text-muted mt-2">
              Repeated for {plasticity_cfg.ns_steps} iterations per update
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper components
function DiagramBox({
  label,
  sublabel,
  color,
  plastic = false,
  highlight = false,
}: {
  label: string;
  sublabel: string;
  color: string;
  plastic?: boolean;
  highlight?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`
        px-4 py-3 rounded-lg text-center min-w-[80px]
        ${plastic ? 'border-2 border-dashed' : 'bg-surface-100'}
        ${highlight ? 'ring-2 ring-accent-green ring-offset-2 ring-offset-surface-50' : ''}
      `}
      style={{
        borderColor: plastic ? color : undefined,
        backgroundColor: plastic ? `${color}10` : undefined,
      }}
    >
      <div className="font-mono text-sm font-bold" style={{ color }}>
        {label}
      </div>
      <div className="text-xs text-text-muted">{sublabel}</div>
    </motion.div>
  );
}

function Arrow() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" className="text-text-muted flex-shrink-0">
      <path
        d="M5 12h14m-4-4l4 4-4 4"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
