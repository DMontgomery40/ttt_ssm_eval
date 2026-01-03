import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { getStatusColor, getStatusLabel, formatRelativeTime } from '../../utils/formatting';
import type { GlobalUpdateEvent, RunData } from '../../types';

interface GlobalUpdateTimelineProps {
  events: GlobalUpdateEvent[];
  runs: RunData[];
}

export function GlobalUpdateTimeline({ events, runs }: GlobalUpdateTimelineProps) {
  const { setSelectedUpdateEvent, selectedRunId, setSelectedRunId } = useDashboardStore();

  // Group events by run
  const runGroups = useMemo(() => {
    const groups = new Map<string, GlobalUpdateEvent[]>();
    for (const event of events) {
      const existing = groups.get(event.run_id) || [];
      existing.push(event);
      groups.set(event.run_id, existing);
    }
    return groups;
  }, [events]);

  // Sort runs by creation time
  const sortedRuns = useMemo(() => {
    return [...runs].sort((a, b) => a.created_at_unix - b.created_at_unix);
  }, [runs]);

  // Stats across all runs
  const globalStats = useMemo(() => {
    const commits = events.filter(e => e.status === 'commit').length;
    const rollbacks = events.filter(e => e.status !== 'commit').length;
    return { commits, rollbacks, total: events.length };
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-text-primary mb-4">Global Update Timeline</h3>
        <div className="text-center text-text-muted py-8">
          No update events recorded across runs.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Global Update Timeline</h3>
          <p className="text-xs text-text-muted mt-1">
            All update events across {runs.length} runs
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <span className="text-accent-green">
            {globalStats.commits} total commits
          </span>
          <span className="text-accent-red">
            {globalStats.rollbacks} total rollbacks
          </span>
        </div>
      </div>

      {/* Run segments */}
      <div className="space-y-2">
        {sortedRuns.map((run, runIdx) => {
          const runEvents = runGroups.get(run.run_id) || [];
          const isSelected = selectedRunId === run.run_id;
          const commits = runEvents.filter(e => e.status === 'commit').length;
          const rollbacks = runEvents.filter(e => e.status !== 'commit').length;

          return (
            <motion.div
              key={run.run_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: runIdx * 0.05 }}
              className={`relative rounded-lg border transition-all ${
                isSelected
                  ? 'border-accent-blue bg-accent-blue/5'
                  : 'border-surface-200 hover:border-surface-300'
              }`}
            >
              {/* Run header */}
              <button
                onClick={() => setSelectedRunId(isSelected ? null : run.run_id)}
                className="w-full flex items-center justify-between p-2 text-left"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${runIdx === sortedRuns.length - 1 ? 'bg-accent-green' : 'bg-surface-300'}`} />
                  <span className="font-mono text-xs text-text-primary">
                    {run.run_id.split('_').slice(-1)[0]}
                  </span>
                  <span className="text-xs text-text-muted">
                    seed={run.seed}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="text-text-muted">
                    {formatRelativeTime(run.created_at_unix)}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-accent-green">{commits}</span>
                    <span className="text-text-muted">/</span>
                    <span className="text-accent-red">{rollbacks}</span>
                  </div>
                </div>
              </button>

              {/* Timeline track */}
              <div className="px-2 pb-2">
                <div className="relative h-6">
                  {/* Background track */}
                  <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-surface-200 -translate-y-1/2" />

                  {/* Event markers */}
                  {runEvents.map((event, i) => {
                    const x = (event.t / run.steps) * 100;
                    const color = getStatusColor(event.status);

                    return (
                      <motion.button
                        key={i}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: runIdx * 0.05 + i * 0.01 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedUpdateEvent(event);
                        }}
                        className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
                        style={{ left: `${x}%` }}
                        title={`t=${event.t}: ${getStatusLabel(event.status)}`}
                      >
                        <div
                          className={`w-2 h-2 rounded-full transition-transform hover:scale-150`}
                          style={{ backgroundColor: color }}
                        />
                      </motion.button>
                    );
                  })}
                </div>

                {/* Time markers */}
                <div className="flex justify-between mt-1 text-xs text-text-muted">
                  <span>t=0</span>
                  <span>t={run.steps}</span>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-surface-200 flex items-center justify-center gap-6 text-xs text-text-muted">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-green" />
          <span>Committed</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-red" />
          <span>Loss Rollback</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-orange" />
          <span>Grad Rollback</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-purple" />
          <span>State Rollback</span>
        </div>
      </div>

      {/* Insight */}
      <div className="mt-4 p-3 bg-accent-blue/10 border border-accent-blue/30 rounded text-xs text-text-secondary">
        <strong className="text-accent-blue">Long-term Learning:</strong> This timeline shows how the model's
        update behavior evolved across runs. Patterns like increasing commit rates or fewer rollbacks indicate
        stable learning. Runs with many rollbacks may need hyperparameter adjustment.
      </div>
    </div>
  );
}
