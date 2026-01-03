import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { WeightHeatmap } from '../charts/WeightHeatmap';
import { WeightDiffHeatmap } from '../charts/WeightDiffHeatmap';
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
  const { currentSession, weightCompareMode, setWeightCompareMode } = useDashboardStore();
  const { weights, parentWeights, baseWeights, meta } = currentSession;

  if (!weights) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted">
        No weight data available for this session.
      </div>
    );
  }

  // Compute singular values (simple approximation for visualization)
  const computeSVD = (matrix: number[][]) => {
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

  // Compute weight difference
  const computeDiff = (current: number[][], reference: number[][] | undefined): number[][] | null => {
    if (!reference) return null;
    if (current.length !== reference.length || current[0]?.length !== reference[0]?.length) return null;

    return current.map((row, i) =>
      row.map((val, j) => val - reference[i][j])
    );
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

  // Determine what weights to compare against
  const referenceWeights = weightCompareMode === 'parent' ? parentWeights :
                           weightCompareMode === 'base' ? baseWeights : null;
  const referenceLabel = weightCompareMode === 'parent' ? 'Parent Session' :
                         weightCompareMode === 'base' ? 'Base Checkpoint' : null;

  // Compute diffs if in comparison mode
  const wuDiff = useMemo(() => computeDiff(weights.W_u, referenceWeights?.W_u), [weights.W_u, referenceWeights]);
  const bDiff = useMemo(() => computeDiff(weights.B, referenceWeights?.B), [weights.B, referenceWeights]);
  const woDiff = useMemo(() => computeDiff(weights.W_o, referenceWeights?.W_o), [weights.W_o, referenceWeights]);

  // Compute drift metrics
  const computeDriftMetrics = (diff: number[][] | null) => {
    if (!diff) return null;
    const flat = diff.flat();
    const maxAbs = Math.max(...flat.map(Math.abs));
    const l2 = Math.sqrt(flat.reduce((sum, v) => sum + v * v, 0));
    const meanAbs = flat.reduce((sum, v) => sum + Math.abs(v), 0) / flat.length;
    return { maxAbs, l2, meanAbs };
  };

  const driftMetrics = {
    W_u: computeDriftMetrics(wuDiff),
    B: computeDriftMetrics(bDiff),
    W_o: computeDriftMetrics(woDiff),
  };

  return (
    <div className="space-y-6">
      {/* Comparison Mode Toggle */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Weight Comparison Mode</h3>
            <p className="text-xs text-text-muted mt-1">
              Compare current weights against parent session or base checkpoint
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setWeightCompareMode('none')}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                weightCompareMode === 'none'
                  ? 'bg-accent-blue text-white'
                  : 'bg-surface-200 text-text-secondary hover:bg-surface-300'
              }`}
            >
              Current Only
            </button>
            <button
              onClick={() => setWeightCompareMode('parent')}
              disabled={!parentWeights}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                weightCompareMode === 'parent'
                  ? 'bg-accent-purple text-white'
                  : 'bg-surface-200 text-text-secondary hover:bg-surface-300'
              } ${!parentWeights ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              vs Parent
            </button>
            <button
              onClick={() => setWeightCompareMode('base')}
              disabled={!baseWeights}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                weightCompareMode === 'base'
                  ? 'bg-accent-gold text-white'
                  : 'bg-surface-200 text-text-secondary hover:bg-surface-300'
              } ${!baseWeights ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              vs Base
            </button>
          </div>
        </div>

        {/* Drift summary when in comparison mode */}
        {weightCompareMode !== 'none' && referenceWeights && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-4 pt-4 border-t border-surface-200"
          >
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm text-text-primary font-medium">
                Weight Drift from {referenceLabel}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(driftMetrics).map(([name, metrics]) => (
                <div key={name} className="bg-surface-100 rounded-lg p-3">
                  <div className="font-mono text-sm mb-2">{name}</div>
                  {metrics ? (
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <span className="text-text-muted block">Max |δ|</span>
                        <span className="font-mono text-accent-red">{metrics.maxAbs.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="text-text-muted block">L2 Drift</span>
                        <span className="font-mono text-accent-purple">{metrics.l2.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="text-text-muted block">Mean |δ|</span>
                        <span className="font-mono text-accent-blue">{metrics.meanAbs.toFixed(4)}</span>
                      </div>
                    </div>
                  ) : (
                    <span className="text-xs text-text-muted">N/A</span>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>

      {/* Weight Matrix Heatmaps */}
      {weightCompareMode === 'none' ? (
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
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            {wuDiff && (
              <WeightDiffHeatmap
                name="W_u Δ"
                data={wuDiff}
                description={`Diff from ${referenceLabel}`}
              />
            )}
            {bDiff && (
              <WeightDiffHeatmap
                name="B Δ"
                data={bDiff}
                description={`Diff from ${referenceLabel}`}
              />
            )}
            {woDiff && (
              <WeightDiffHeatmap
                name="W_o Δ"
                data={woDiff}
                description={`Diff from ${referenceLabel}`}
              />
            )}
          </div>

          {/* Current weights for reference */}
          <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Current Weights (Reference)</h3>
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
          </div>
        </div>
      )}

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

      {/* Learning insight */}
      <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-4">
        <p className="text-sm text-text-primary">
          <strong className="text-accent-blue">Phase 1 Insight:</strong> Use the comparison modes above to see
          how much the weights have drifted from the parent session (what this fork learned) or from the base
          checkpoint (what the entire lineage has learned). Large drifts in specific regions indicate where the
          model adapted most to the physics environment.
        </p>
      </div>
    </div>
  );
}
