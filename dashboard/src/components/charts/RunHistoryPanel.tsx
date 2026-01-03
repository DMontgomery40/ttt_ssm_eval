import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { formatRelativeTime } from '../../utils/formatting';
import type { RunData } from '../../types';

interface RunHistoryPanelProps {
  runs: RunData[];
}

export function RunHistoryPanel({ runs }: RunHistoryPanelProps) {
  const { selectedRunId, setSelectedRunId } = useDashboardStore();
  const [isExpanded, setIsExpanded] = useState(false);

  // Sort runs by time (newest first for display)
  const sortedRuns = [...runs].sort((a, b) => b.created_at_unix - a.created_at_unix);

  // Currently selected run (or latest)
  const currentRun = selectedRunId
    ? runs.find(r => r.run_id === selectedRunId)
    : sortedRuns[0];

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-text-primary">Run History</h3>
          <span className="text-xs px-2 py-0.5 bg-surface-200 rounded-full text-text-muted">
            {runs.length} runs
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-accent-blue hover:underline"
        >
          {isExpanded ? 'Collapse' : 'Expand all'}
        </button>
      </div>

      {/* Current run highlight */}
      {currentRun && (
        <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-3 mb-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-accent-blue">Currently viewing:</span>
              <div className="font-mono text-sm text-text-primary mt-1">
                {currentRun.run_id}
              </div>
            </div>
            <div className="text-right text-xs">
              <div className="text-text-muted">
                {formatRelativeTime(currentRun.created_at_unix)}
              </div>
              <div className="mt-1">
                <span className="text-accent-green">{currentRun.metrics.updates_committed} commits</span>
                {currentRun.metrics.updates_rolled_back > 0 && (
                  <span className="text-accent-red ml-2">{currentRun.metrics.updates_rolled_back} rollbacks</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Run list */}
      <div className="space-y-2">
        <AnimatePresence>
          {(isExpanded ? sortedRuns : sortedRuns.slice(0, 3)).map((run, idx) => (
            <motion.button
              key={run.run_id}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ delay: idx * 0.05 }}
              onClick={() => setSelectedRunId(run.run_id === selectedRunId ? null : run.run_id)}
              className={`
                w-full flex items-center justify-between p-3 rounded-lg border transition-all
                ${run.run_id === (selectedRunId || sortedRuns[0]?.run_id)
                  ? 'border-accent-blue bg-surface-100'
                  : 'border-surface-200 bg-surface hover:bg-surface-100'
                }
              `}
            >
              <div className="flex items-center gap-3">
                {/* Run indicator */}
                <div className={`
                  w-2 h-2 rounded-full
                  ${idx === 0 ? 'bg-accent-green' : 'bg-surface-300'}
                `} />

                <div className="text-left">
                  <div className="font-mono text-xs text-text-primary">
                    {run.run_id.split('_').slice(-1)[0]}
                  </div>
                  <div className="text-xs text-text-muted">
                    seed={run.seed}, {run.steps} steps
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4 text-xs">
                <div className="text-right">
                  <div className="text-text-muted">
                    {formatRelativeTime(run.created_at_unix)}
                  </div>
                </div>

                {/* Update stats */}
                <div className="flex items-center gap-2">
                  <span className="px-1.5 py-0.5 bg-accent-green/20 text-accent-green rounded">
                    {run.metrics.updates_committed}
                  </span>
                  {run.metrics.updates_rolled_back > 0 && (
                    <span className="px-1.5 py-0.5 bg-accent-red/20 text-accent-red rounded">
                      {run.metrics.updates_rolled_back}
                    </span>
                  )}
                </div>
              </div>
            </motion.button>
          ))}
        </AnimatePresence>
      </div>

      {/* Show more indicator */}
      {!isExpanded && runs.length > 3 && (
        <button
          onClick={() => setIsExpanded(true)}
          className="w-full mt-2 py-2 text-xs text-text-muted hover:text-text-secondary transition-colors"
        >
          + {runs.length - 3} more runs
        </button>
      )}
    </div>
  );
}
