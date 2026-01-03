import { useMemo } from 'react';
import { useDashboardStore } from '../../store';
import { WeightHeatmap } from '../charts/WeightHeatmap';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export function WeightsTab() {
  const { currentSession } = useDashboardStore();
  const { weights, meta } = currentSession;

  if (!weights) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted">
        No weight data available for this session.
      </div>
    );
  }

  // Compute singular values (simple approximation for visualization)
  const computeSVD = (matrix: number[][]) => {
    // For a proper SVD we'd need a library, but for visualization we'll compute column norms
    const cols = matrix[0]?.length ?? 0;
    const svs: number[] = [];

    for (let j = 0; j < cols; j++) {
      let norm = 0;
      for (let i = 0; i < matrix.length; i++) {
        norm += matrix[i][j] ** 2;
      }
      svs.push(Math.sqrt(norm));
    }

    return svs.sort((a, b) => b - a).slice(0, 10);
  };

  const wuSVs = useMemo(() => computeSVD(weights.W_u), [weights.W_u]);
  const bSVs = useMemo(() => computeSVD(weights.B), [weights.B]);
  const woSVs = useMemo(() => computeSVD(weights.W_o), [weights.W_o]);

  const svData = wuSVs.map((_, i) => ({
    idx: i + 1,
    W_u: wuSVs[i] ?? 0,
    B: bSVs[i] ?? 0,
    W_o: woSVs[i] ?? 0,
  }));

  // Parameter counts
  const paramCounts = [
    { name: 'W_u', count: meta.model_cfg.u_dim * (meta.model_cfg.z_dim + meta.model_cfg.act_dim), plastic: true },
    { name: 'B', count: meta.model_cfg.n_state * meta.model_cfg.u_dim, plastic: true },
    { name: 'W_o', count: meta.model_cfg.z_dim * meta.model_cfg.n_state, plastic: true },
    { name: 'a_raw', count: meta.model_cfg.n_state, plastic: false },
  ];

  const totalPlastic = paramCounts.filter(p => p.plastic).reduce((a, b) => a + b.count, 0);
  const totalFrozen = paramCounts.filter(p => !p.plastic).reduce((a, b) => a + b.count, 0);

  return (
    <div className="space-y-6">
      {/* Weight Matrix Heatmaps */}
      <div className="grid grid-cols-3 gap-4">
        <WeightHeatmap
          name="W_u"
          data={weights.W_u}
          description="Input projection"
        />
        <WeightHeatmap
          name="B"
          data={weights.B}
          description="State update"
        />
        <WeightHeatmap
          name="W_o"
          data={weights.W_o}
          description="Output projection"
        />
      </div>

      {/* Spectral View */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-4">
          Column Norms (Spectral Proxy)
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={svData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="idx" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 11 }} />
            <YAxis stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 11 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#161b22',
                border: '1px solid #30363d',
                borderRadius: '8px',
              }}
              labelStyle={{ color: '#8b949e' }}
            />
            <Bar dataKey="W_u" fill="#58a6ff" name="W_u" />
            <Bar dataKey="B" fill="#39d353" name="B" />
            <Bar dataKey="W_o" fill="#a371f7" name="W_o" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Parameter Count Summary */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Parameter Count</h3>
        <div className="grid grid-cols-4 gap-4">
          {paramCounts.map((p) => (
            <div
              key={p.name}
              className={`p-3 rounded-lg ${p.plastic ? 'bg-accent-green/10' : 'bg-surface-100'}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-sm">{p.name}</span>
                <span
                  className={`text-xs px-1.5 py-0.5 rounded ${
                    p.plastic ? 'bg-accent-green/20 text-accent-green' : 'bg-surface-200 text-text-muted'
                  }`}
                >
                  {p.plastic ? 'plastic' : 'frozen'}
                </span>
              </div>
              <span className="text-lg font-bold font-mono text-text-primary">
                {p.count.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-8 text-sm">
          <div>
            <span className="text-text-muted">Total Plastic: </span>
            <span className="font-mono text-accent-green font-bold">{totalPlastic.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-text-muted">Total Frozen: </span>
            <span className="font-mono text-text-secondary font-bold">{totalFrozen.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-text-muted">Plasticity Ratio: </span>
            <span className="font-mono text-accent-blue font-bold">
              {((totalPlastic / (totalPlastic + totalFrozen)) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
