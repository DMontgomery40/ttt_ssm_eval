import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { formatRelativeTime, truncateHash, formatImprovement } from '../../utils/formatting';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
} from 'recharts';

const SESSION_COLORS = ['#58a6ff', '#39d353', '#a371f7', '#f0883e', '#da3633'];

export function SessionsTab() {
  const { sessions, currentSession, selectedSessionIds, toggleSessionSelection, setCurrentSession } =
    useDashboardStore();

  // Comparison data
  const comparisonData = useMemo(() => {
    if (selectedSessionIds.length < 2) return null;

    const selectedSessions = sessions.filter((s) => selectedSessionIds.includes(s.meta.session_id));

    // Align by timestep for line chart
    const maxSteps = Math.max(...selectedSessions.map((s) => s.perStep.length));
    const lineData = Array.from({ length: maxSteps }, (_, t) => {
      const point: Record<string, number> = { t };
      selectedSessions.forEach((session) => {
        const step = session.perStep[t];
        if (step) {
          point[session.meta.session_id] = step.adaptive_mse;
        }
      });
      return point;
    });

    // Bar chart data for final metrics
    const barData = selectedSessions.map((s) => ({
      session_id: s.meta.session_id,
      mu: s.meta.mu,
      baseline: s.metrics.baseline_mse_last100_mean,
      adaptive: s.metrics.adaptive_mse_last100_mean,
      improvement:
        ((s.metrics.baseline_mse_last100_mean - s.metrics.adaptive_mse_last100_mean) /
          s.metrics.baseline_mse_last100_mean) *
        100,
    }));

    return { lineData, barData, sessions: selectedSessions };
  }, [sessions, selectedSessionIds]);

  return (
    <div className="space-y-6">
      {/* Session List */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg">
        <div className="p-4 border-b border-surface-200">
          <h3 className="text-sm font-semibold text-text-primary">Available Sessions</h3>
          <p className="text-xs text-text-muted mt-1">
            Select 2+ sessions to compare, or click to load
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-surface-100 text-text-muted text-xs uppercase">
              <tr>
                <th className="px-4 py-3 text-left w-8">
                  <input
                    type="checkbox"
                    checked={selectedSessionIds.length === sessions.length}
                    onChange={() => {
                      if (selectedSessionIds.length === sessions.length) {
                        sessions.forEach((s) => toggleSessionSelection(s.meta.session_id));
                      } else {
                        sessions
                          .filter((s) => !selectedSessionIds.includes(s.meta.session_id))
                          .forEach((s) => toggleSessionSelection(s.meta.session_id));
                      }
                    }}
                    className="rounded"
                  />
                </th>
                <th className="px-4 py-3 text-left">Session ID</th>
                <th className="px-4 py-3 text-left">μ Value</th>
                <th className="px-4 py-3 text-left">Mode</th>
                <th className="px-4 py-3 text-right">Commits</th>
                <th className="px-4 py-3 text-right">Improvement</th>
                <th className="px-4 py-3 text-left">Created</th>
                <th className="px-4 py-3 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-200">
              {sessions.map((session, idx) => {
                const isSelected = selectedSessionIds.includes(session.meta.session_id);
                const isCurrent = currentSession.meta.session_id === session.meta.session_id;

                return (
                  <motion.tr
                    key={session.meta.session_id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: idx * 0.05 }}
                    className={`
                      transition-colors
                      ${isCurrent ? 'bg-accent-blue/10' : 'hover:bg-surface-100'}
                    `}
                  >
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleSessionSelection(session.meta.session_id)}
                        className="rounded"
                      />
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-mono">{session.meta.session_id}</span>
                      {isCurrent && (
                        <span className="ml-2 text-xs bg-accent-blue/20 text-accent-blue px-1.5 py-0.5 rounded">
                          current
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-mono text-accent-purple">{session.meta.mu.toFixed(4)}</span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`text-xs px-1.5 py-0.5 rounded ${
                          session.meta.env_mode === 'linear'
                            ? 'bg-accent-green/20 text-accent-green'
                            : 'bg-accent-purple/20 text-accent-purple'
                        }`}
                      >
                        {session.meta.env_mode}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {session.metrics.updates_committed} / {session.metrics.updates_attempted}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="font-mono text-accent-green">
                        {formatImprovement(
                          session.metrics.baseline_mse_last100_mean,
                          session.metrics.adaptive_mse_last100_mean
                        )}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-text-muted">
                      {formatRelativeTime(session.meta.created_at_unix)}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => setCurrentSession(session)}
                        disabled={isCurrent}
                        className={`px-2 py-1 text-xs rounded transition-colors ${
                          isCurrent
                            ? 'bg-surface-200 text-text-muted cursor-not-allowed'
                            : 'bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30'
                        }`}
                      >
                        {isCurrent ? 'Loaded' : 'Load'}
                      </button>
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison Charts */}
      {comparisonData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">
              MSE Comparison ({selectedSessionIds.length} sessions)
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={comparisonData.lineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="t" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <YAxis stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                {comparisonData.sessions.map((s, i) => (
                  <Line
                    key={s.meta.session_id}
                    type="monotone"
                    dataKey={s.meta.session_id}
                    stroke={SESSION_COLORS[i % SESSION_COLORS.length]}
                    strokeWidth={1.5}
                    dot={false}
                    name={`${s.meta.session_id} (μ=${s.meta.mu.toFixed(3)})`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Final Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={comparisonData.barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="session_id" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <YAxis stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Bar dataKey="baseline" fill="#6e7681" name="Baseline MSE" />
                <Bar dataKey="adaptive" fill="#39d353" name="Adaptive MSE" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Current Session Resume Info */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Session Resume Info</h3>

        <div className="bg-surface-100 rounded-lg p-4 font-mono text-sm">
          <p className="text-text-muted mb-2">Resume this session with:</p>
          <code className="block bg-surface rounded p-2 text-accent-blue break-all">
            python phase0_muon.py --resume_session_dir=./runs/{currentSession.meta.session_id}/session
          </code>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <span className="text-xs text-text-muted block mb-1">Model Signature</span>
            <code className="text-xs font-mono text-text-secondary break-all">
              {truncateHash(currentSession.meta.model_signature, 16)}
            </code>
          </div>
          <div>
            <span className="text-xs text-text-muted block mb-1">Base Checkpoint Hash</span>
            <code className="text-xs font-mono text-text-secondary break-all">
              {truncateHash(currentSession.meta.base_ckpt_hash, 16)}
            </code>
          </div>
        </div>

        <div className="mt-4 p-3 bg-accent-blue/10 border border-accent-blue/30 rounded text-xs text-text-secondary">
          <strong className="text-accent-blue">Note:</strong> All sessions with matching model_signature
          are compatible for comparison. The signature ensures identical model architecture and base weights.
        </div>
      </div>
    </div>
  );
}
