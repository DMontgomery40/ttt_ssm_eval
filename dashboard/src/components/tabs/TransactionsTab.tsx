import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { getStatusColor, getStatusLabel, formatNumber } from '../../utils/formatting';
import { GlobalUpdateTimeline } from '../charts/GlobalUpdateTimeline';
import type { UpdateStatus } from '../../types';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

type SortKey = 't' | 'status' | 'pre_loss' | 'post_loss' | 'grad_norm';
type SortDir = 'asc' | 'desc';

export function TransactionsTab() {
  const { currentSession, selectedUpdateEvent, setSelectedUpdateEvent } = useDashboardStore();
  const { updateEvents, globalUpdateEvents, runs, meta } = currentSession;

  const [sortKey, setSortKey] = useState<SortKey>('t');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [statusFilter, setStatusFilter] = useState<UpdateStatus | 'all'>('all');

  // Sort and filter events
  const filteredEvents = useMemo(() => {
    let events = [...updateEvents];

    if (statusFilter !== 'all') {
      events = events.filter((e) => e.status === statusFilter);
    }

    events.sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      switch (sortKey) {
        case 't':
          aVal = a.t;
          bVal = b.t;
          break;
        case 'status':
          aVal = a.status;
          bVal = b.status;
          break;
        case 'pre_loss':
          aVal = a.pre_loss;
          bVal = b.pre_loss;
          break;
        case 'post_loss':
          aVal = a.post_loss ?? Infinity;
          bVal = b.post_loss ?? Infinity;
          break;
        case 'grad_norm':
          aVal = a.grad_norm;
          bVal = b.grad_norm;
          break;
        default:
          return 0;
      }

      if (typeof aVal === 'string') {
        return sortDir === 'asc' ? aVal.localeCompare(bVal as string) : (bVal as string).localeCompare(aVal);
      }
      return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    });

    return events;
  }, [updateEvents, sortKey, sortDir, statusFilter]);

  // Rollback breakdown
  const rollbackBreakdown = useMemo(() => {
    const counts = {
      commit: 0,
      rollback_loss_regression: 0,
      rollback_grad_norm: 0,
      rollback_state_norm: 0,
    };

    for (const e of updateEvents) {
      counts[e.status]++;
    }

    return [
      { name: 'Committed', value: counts.commit, color: '#238636' },
      { name: 'Loss Regression', value: counts.rollback_loss_regression, color: '#da3633' },
      { name: 'Grad Explosion', value: counts.rollback_grad_norm, color: '#d29922' },
      { name: 'State Explosion', value: counts.rollback_state_norm, color: '#a371f7' },
    ].filter((d) => d.value > 0);
  }, [updateEvents]);

  // Threshold lines data
  const thresholdData = updateEvents.map((e) => ({
    t: e.t,
    grad_norm: e.grad_norm,
    threshold: meta.plasticity_cfg.grad_norm_max,
  }));

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const SortHeader = ({ label, sortKey: key }: { label: string; sortKey: SortKey }) => (
    <button
      onClick={() => handleSort(key)}
      className="flex items-center gap-1 hover:text-accent-blue transition-colors"
    >
      {label}
      {sortKey === key && (
        <span className="text-accent-blue">{sortDir === 'asc' ? '↑' : '↓'}</span>
      )}
    </button>
  );

  return (
    <div className="space-y-6">
      {/* Global Update Timeline (Phase 1) */}
      {runs && runs.length > 0 && globalUpdateEvents && (
        <GlobalUpdateTimeline events={globalUpdateEvents} runs={runs} />
      )}

      {/* Rollback Analysis */}
      <div className="grid grid-cols-2 gap-4">
        {/* Pie chart */}
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Update Outcomes</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={rollbackBreakdown}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={2}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
                labelLine={{ stroke: '#6e7681' }}
              >
                {rollbackBreakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Gradient norm vs threshold */}
        <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Gradient Norm vs Threshold</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={thresholdData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis dataKey="t" stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 10 }} />
              <YAxis stroke="#6e7681" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="grad_norm" fill="#58a6ff" name="Grad Norm" />
              <Bar dataKey="threshold" fill="#da363366" name="Threshold" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Transaction Table */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg">
        {/* Table header */}
        <div className="flex items-center justify-between p-4 border-b border-surface-200">
          <h3 className="text-sm font-semibold text-text-primary">Transaction Log</h3>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as UpdateStatus | 'all')}
            className="bg-surface-100 border border-surface-200 rounded px-2 py-1 text-sm"
          >
            <option value="all">All statuses</option>
            <option value="commit">Committed only</option>
            <option value="rollback_loss_regression">Loss rollbacks</option>
            <option value="rollback_grad_norm">Grad rollbacks</option>
          </select>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-surface-100 text-text-muted text-xs uppercase">
              <tr>
                <th className="px-4 py-3 text-left"><SortHeader label="Timestep" sortKey="t" /></th>
                <th className="px-4 py-3 text-left"><SortHeader label="Status" sortKey="status" /></th>
                <th className="px-4 py-3 text-right"><SortHeader label="Pre-loss" sortKey="pre_loss" /></th>
                <th className="px-4 py-3 text-right"><SortHeader label="Post-loss" sortKey="post_loss" /></th>
                <th className="px-4 py-3 text-right">Δ Loss</th>
                <th className="px-4 py-3 text-right"><SortHeader label="Grad Norm" sortKey="grad_norm" /></th>
                <th className="px-4 py-3 text-right">Max |h|</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-200">
              {filteredEvents.map((event) => {
                const deltaLoss = event.post_loss ? event.post_loss - event.pre_loss : null;
                const isSelected = selectedUpdateEvent?.t === event.t;

                return (
                  <motion.tr
                    key={event.t}
                    onClick={() => setSelectedUpdateEvent(isSelected ? null : event)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'selected-row' : 'hover:bg-surface-100'
                    }`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <td className="px-4 py-3 font-mono">{event.t}</td>
                    <td className="px-4 py-3">
                      <span
                        className="inline-block px-2 py-0.5 rounded text-xs font-medium"
                        style={{
                          backgroundColor: `${getStatusColor(event.status)}20`,
                          color: getStatusColor(event.status),
                        }}
                      >
                        {getStatusLabel(event.status)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{formatNumber(event.pre_loss)}</td>
                    <td className="px-4 py-3 text-right font-mono">
                      {event.post_loss ? formatNumber(event.post_loss) : '—'}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {deltaLoss !== null ? (
                        <span className={deltaLoss < 0 ? 'text-accent-green' : 'text-accent-red'}>
                          {deltaLoss >= 0 ? '+' : ''}{formatNumber(deltaLoss)}
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      <span className={event.grad_norm > meta.plasticity_cfg.grad_norm_max ? 'text-accent-red' : ''}>
                        {event.grad_norm.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {event.post_max_h?.toFixed(2) ?? event.pre_max_h.toFixed(2)}
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {filteredEvents.length === 0 && (
          <div className="p-8 text-center text-text-muted">
            No transactions match the current filter.
          </div>
        )}
      </div>

      {/* Detail Panel */}
      <AnimatePresence>
        {selectedUpdateEvent && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="bg-surface-50 border border-surface-200 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-text-primary">
                Transaction Details — t={selectedUpdateEvent.t}
              </h3>
              <button
                onClick={() => setSelectedUpdateEvent(null)}
                className="text-text-muted hover:text-text-primary"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-surface-100 rounded p-3">
                <span className="text-xs text-text-muted block mb-1">Pre → Post Loss</span>
                <div className="flex items-center gap-2 font-mono">
                  <span>{formatNumber(selectedUpdateEvent.pre_loss)}</span>
                  <span className="text-text-muted">→</span>
                  <span
                    className={
                      selectedUpdateEvent.post_loss
                        ? selectedUpdateEvent.post_loss < selectedUpdateEvent.pre_loss
                          ? 'text-accent-green'
                          : 'text-accent-red'
                        : 'text-text-muted'
                    }
                  >
                    {selectedUpdateEvent.post_loss ? formatNumber(selectedUpdateEvent.post_loss) : 'N/A'}
                  </span>
                </div>
              </div>
              <div className="bg-surface-100 rounded p-3">
                <span className="text-xs text-text-muted block mb-1">Gradient Norm</span>
                <span
                  className={`font-mono ${
                    selectedUpdateEvent.grad_norm > meta.plasticity_cfg.grad_norm_max
                      ? 'text-accent-red'
                      : 'text-text-primary'
                  }`}
                >
                  {selectedUpdateEvent.grad_norm.toFixed(4)}
                </span>
                <span className="text-xs text-text-muted ml-2">
                  / {meta.plasticity_cfg.grad_norm_max} max
                </span>
              </div>
              <div className="bg-surface-100 rounded p-3">
                <span className="text-xs text-text-muted block mb-1">Hidden State Norm</span>
                <span className="font-mono">
                  {selectedUpdateEvent.pre_max_h.toFixed(4)}
                  {selectedUpdateEvent.post_max_h && (
                    <>
                      <span className="text-text-muted"> → </span>
                      {selectedUpdateEvent.post_max_h.toFixed(4)}
                    </>
                  )}
                </span>
              </div>
            </div>

            {/* Explanation */}
            <div
              className="p-3 rounded text-sm"
              style={{
                backgroundColor: `${getStatusColor(selectedUpdateEvent.status)}10`,
                borderLeft: `3px solid ${getStatusColor(selectedUpdateEvent.status)}`,
              }}
            >
              {selectedUpdateEvent.status === 'commit' && selectedUpdateEvent.post_loss && (
                <>
                  <span className="text-accent-green font-medium">Update committed successfully.</span>
                  <span className="text-text-secondary ml-1">
                    Loss improved by{' '}
                    {(
                      ((selectedUpdateEvent.pre_loss - selectedUpdateEvent.post_loss) /
                        selectedUpdateEvent.pre_loss) *
                      100
                    ).toFixed(1)}
                    % and all safety checks passed.
                  </span>
                </>
              )}
              {selectedUpdateEvent.status === 'rollback_loss_regression' && (
                <>
                  <span className="text-accent-red font-medium">Rolled back due to loss regression.</span>
                  <span className="text-text-secondary ml-1">
                    Post-loss ({formatNumber(selectedUpdateEvent.post_loss ?? 0)}) exceeded tolerance of{' '}
                    {((1 + meta.plasticity_cfg.rollback_tol) * 100).toFixed(0)}% of pre-loss (
                    {formatNumber(selectedUpdateEvent.pre_loss * (1 + meta.plasticity_cfg.rollback_tol))}).
                  </span>
                </>
              )}
              {selectedUpdateEvent.status === 'rollback_grad_norm' && (
                <>
                  <span className="text-accent-orange font-medium">Rolled back due to gradient explosion.</span>
                  <span className="text-text-secondary ml-1">
                    Gradient norm ({selectedUpdateEvent.grad_norm.toFixed(2)}) exceeded threshold (
                    {meta.plasticity_cfg.grad_norm_max}).
                  </span>
                </>
              )}
              {selectedUpdateEvent.status === 'rollback_state_norm' && (
                <>
                  <span className="text-accent-purple font-medium">Rolled back due to state explosion.</span>
                  <span className="text-text-secondary ml-1">
                    Hidden state norm exceeded threshold ({meta.plasticity_cfg.state_norm_max.toExponential(0)}).
                  </span>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
