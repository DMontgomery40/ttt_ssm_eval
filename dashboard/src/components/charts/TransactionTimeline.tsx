import { motion } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { getStatusColor, getStatusLabel } from '../../utils/formatting';
import type { UpdateEvent } from '../../types';

interface TransactionTimelineProps {
  events: UpdateEvent[];
  maxTime: number;
}

export function TransactionTimeline({ events, maxTime }: TransactionTimelineProps) {
  const { selectedUpdateEvent, setSelectedUpdateEvent } = useDashboardStore();

  // Count by status
  const counts = events.reduce(
    (acc, e) => {
      if (e.status === 'commit') acc.commits++;
      else if (e.status === 'rollback_loss_regression') acc.lossRollbacks++;
      else if (e.status === 'rollback_grad_norm') acc.gradRollbacks++;
      else if (e.status === 'rollback_state_norm') acc.stateRollbacks++;
      return acc;
    },
    { commits: 0, lossRollbacks: 0, gradRollbacks: 0, stateRollbacks: 0 }
  );

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-text-primary">Update Transaction Timeline</h3>
        <div className="flex items-center gap-4 text-xs">
          <span className="text-accent-green">
            {counts.commits} commits
          </span>
          {counts.lossRollbacks > 0 && (
            <span className="text-accent-red">
              {counts.lossRollbacks} loss rollbacks
            </span>
          )}
          {counts.gradRollbacks > 0 && (
            <span className="text-accent-orange">
              {counts.gradRollbacks} grad rollbacks
            </span>
          )}
          {counts.stateRollbacks > 0 && (
            <span className="text-accent-purple">
              {counts.stateRollbacks} state rollbacks
            </span>
          )}
        </div>
      </div>

      {/* Timeline */}
      <div className="relative h-16">
        {/* Track */}
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-surface-200 -translate-y-1/2" />

        {/* Nodes */}
        {events.map((event, i) => {
          const x = (event.t / maxTime) * 100;
          const isSelected = selectedUpdateEvent?.t === event.t;
          const color = getStatusColor(event.status);

          return (
            <motion.button
              key={i}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: i * 0.02 }}
              onClick={() => setSelectedUpdateEvent(isSelected ? null : event)}
              className="timeline-node absolute top-1/2 -translate-x-1/2 -translate-y-1/2 z-10"
              style={{ left: `${x}%` }}
              title={`t=${event.t}: ${getStatusLabel(event.status)}`}
            >
              <div
                className={`
                  w-4 h-4 rounded-full border-2 transition-all
                  ${isSelected ? 'ring-2 ring-offset-2 ring-offset-surface-50' : ''}
                `}
                style={{
                  backgroundColor: color,
                  borderColor: isSelected ? '#fff' : color,
                }}
              />
            </motion.button>
          );
        })}
      </div>

      {/* Time markers */}
      <div className="flex justify-between mt-2 text-xs text-text-muted">
        <span>t=0</span>
        <span>t={Math.floor(maxTime / 2)}</span>
        <span>t={maxTime}</span>
      </div>

      {/* Selected event detail */}
      {selectedUpdateEvent && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-surface-100 rounded-lg border border-surface-200"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">
              Update at t={selectedUpdateEvent.t}
            </span>
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded"
              style={{
                backgroundColor: `${getStatusColor(selectedUpdateEvent.status)}20`,
                color: getStatusColor(selectedUpdateEvent.status),
              }}
            >
              {getStatusLabel(selectedUpdateEvent.status)}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4 text-xs">
            <div>
              <span className="text-text-muted block">Pre-loss</span>
              <span className="font-mono text-text-primary">
                {selectedUpdateEvent.pre_loss.toExponential(3)}
              </span>
            </div>
            <div>
              <span className="text-text-muted block">Post-loss</span>
              <span className="font-mono text-text-primary">
                {selectedUpdateEvent.post_loss?.toExponential(3) ?? 'N/A'}
              </span>
            </div>
            <div>
              <span className="text-text-muted block">Grad norm</span>
              <span
                className="font-mono"
                style={{
                  color: selectedUpdateEvent.grad_norm > 20 ? '#da3633' : '#e6edf3',
                }}
              >
                {selectedUpdateEvent.grad_norm.toFixed(2)}
              </span>
            </div>
          </div>

          {selectedUpdateEvent.status !== 'commit' && (
            <div className="mt-3 p-2 bg-surface rounded text-xs">
              <span className="text-accent-red">Rollback reason: </span>
              <span className="text-text-secondary">
                {selectedUpdateEvent.status === 'rollback_loss_regression' &&
                  `Post-loss (${selectedUpdateEvent.post_loss?.toExponential(2)}) exceeded tolerance`}
                {selectedUpdateEvent.status === 'rollback_grad_norm' &&
                  `Gradient norm (${selectedUpdateEvent.grad_norm.toFixed(1)}) exceeded max (20.0)`}
                {selectedUpdateEvent.status === 'rollback_state_norm' &&
                  `Hidden state norm exceeded threshold`}
              </span>
            </div>
          )}

          {selectedUpdateEvent.status === 'commit' && selectedUpdateEvent.post_loss && (
            <div className="mt-3 p-2 bg-surface rounded text-xs">
              <span className="text-accent-green">Improvement: </span>
              <span className="text-text-secondary">
                Loss reduced by{' '}
                {(
                  ((selectedUpdateEvent.pre_loss - selectedUpdateEvent.post_loss) /
                    selectedUpdateEvent.pre_loss) *
                  100
                ).toFixed(1)}
                %
              </span>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
