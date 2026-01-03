import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../../store';
import { MetricCard, ProgressCard } from '../cards/MetricCard';
import { SessionCard, ConfigCard } from '../cards/SessionCard';
import { MSEComparisonChart } from '../charts/MSEComparisonChart';
import { TransactionTimeline } from '../charts/TransactionTimeline';
import { RunHistoryPanel } from '../charts/RunHistoryPanel';
import { formatImprovement } from '../../utils/formatting';

// Lineage breadcrumb component
function LineageBreadcrumb() {
  const { currentSession, sessions, setCurrentSession, setActiveTab } = useDashboardStore();
  const lineage = useDashboardStore((state) => state.getSessionLineage());

  const handleClick = (sessionId: string) => {
    const session = sessions.find(s => s.meta.session_id === sessionId);
    if (session) {
      setCurrentSession(session);
    }
  };

  // Truncate middle if too long
  const displayLineage = useMemo(() => {
    if (lineage.length <= 4) return lineage;
    return [lineage[0], '...', lineage[lineage.length - 2], lineage[lineage.length - 1]];
  }, [lineage]);

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-text-muted">Lineage:</span>
      <button
        onClick={() => setActiveTab('session-tree')}
        className="text-accent-gold hover:underline font-mono"
      >
        base
      </button>
      {displayLineage.map((item, idx) => (
        <span key={idx} className="flex items-center gap-2">
          <span className="text-text-muted">→</span>
          {item === '...' ? (
            <span className="text-text-muted">...</span>
          ) : (
            <button
              onClick={() => handleClick(item)}
              className={`font-mono hover:underline ${
                item === currentSession.meta.session_id
                  ? 'text-accent-blue font-bold'
                  : 'text-text-secondary'
              }`}
            >
              {item}
            </button>
          )}
        </span>
      ))}
    </div>
  );
}

// Learning insights card
function LearningInsightsCard({
  baseMSE,
  sessionStartMSE,
  adaptiveMSE
}: {
  baseMSE: number;
  sessionStartMSE: number;
  adaptiveMSE: number;
}) {
  // Calculate improvements
  const persistentLearning = baseMSE > 0 ? ((baseMSE - sessionStartMSE) / baseMSE) * 100 : 0;
  const onlineLearning = sessionStartMSE > 0 ? ((sessionStartMSE - adaptiveMSE) / sessionStartMSE) * 100 : 0;
  const totalLearning = baseMSE > 0 ? ((baseMSE - adaptiveMSE) / baseMSE) * 100 : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-surface-50 border border-surface-200 rounded-lg p-4"
    >
      <h3 className="text-sm font-semibold text-text-primary mb-4">Learning Breakdown</h3>

      <div className="grid grid-cols-3 gap-4">
        {/* Persistent Learning */}
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-3 h-0.5 bg-accent-blue" style={{ borderStyle: 'dashed' }} />
            <span className="text-xs text-text-muted">Persistent Learning</span>
          </div>
          <div className="text-2xl font-bold font-mono text-accent-blue">
            {persistentLearning >= 0 ? '+' : ''}{persistentLearning.toFixed(1)}%
          </div>
          <div className="text-xs text-text-muted mt-1">
            base → session_start
          </div>
          <div className="mt-2 text-xs text-text-secondary">
            What previous runs learned (accumulated)
          </div>
        </div>

        {/* Online Learning */}
        <div className="bg-surface-100 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-3 h-1 bg-accent-green rounded" />
            <span className="text-xs text-text-muted">Online Learning</span>
          </div>
          <div className="text-2xl font-bold font-mono text-accent-green">
            {onlineLearning >= 0 ? '+' : ''}{onlineLearning.toFixed(1)}%
          </div>
          <div className="text-xs text-text-muted mt-1">
            session_start → adaptive
          </div>
          <div className="mt-2 text-xs text-text-secondary">
            What this run learned (online TTT)
          </div>
        </div>

        {/* Total Learning */}
        <div className="bg-accent-purple/10 border border-accent-purple/30 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-text-muted">Total Learning</span>
          </div>
          <div className="text-2xl font-bold font-mono text-accent-purple">
            {totalLearning >= 0 ? '+' : ''}{totalLearning.toFixed(1)}%
          </div>
          <div className="text-xs text-text-muted mt-1">
            base → adaptive
          </div>
          <div className="mt-2 text-xs text-text-secondary">
            Cumulative improvement from pretrained
          </div>
        </div>
      </div>

      {/* Visual representation */}
      <div className="mt-4 p-3 bg-surface rounded-lg">
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-4 h-px bg-text-muted border-dashed border-b" />
            <span className="text-text-muted">Base (frozen)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-px bg-accent-blue border-dashed border-b" />
            <span className="text-text-muted">Session Start</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-1 bg-accent-green rounded" />
            <span className="text-text-muted">Adaptive (TTT)</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export function OverviewTab() {
  const { currentSession } = useDashboardStore();
  const { meta, metrics, perStep, updateEvents, runs } = currentSession;

  // Sparkline data (last 100 points)
  const baseSparkline = perStep.slice(-100).map((s) => s.base_mse);
  const sessionStartSparkline = perStep.slice(-100).map((s) => s.session_start_mse);
  const adaptiveSparkline = perStep.slice(-100).map((s) => s.adaptive_mse);

  const totalImprovement = formatImprovement(
    metrics.base_mse_last100_mean,
    metrics.adaptive_last100_mean
  );

  const maxTime = perStep.length > 0 ? perStep[perStep.length - 1].t : 0;

  return (
    <div className="space-y-6">
      {/* Lineage Breadcrumb */}
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-3">
        <LineageBreadcrumb />
      </div>

      {/* Hero Metrics Row */}
      <div className="grid grid-cols-6 gap-4">
        <MetricCard
          title="Base MSE (last 100)"
          value={metrics.base_mse_last100_mean}
          sparklineData={baseSparkline}
          color="#6e7681"
          icon="📉"
        />
        <MetricCard
          title="Session Start MSE"
          value={metrics.session_no_update_last100_mean}
          sparklineData={sessionStartSparkline}
          color="#58a6ff"
          icon="📊"
        />
        <MetricCard
          title="Adaptive MSE"
          value={metrics.adaptive_last100_mean}
          sparklineData={adaptiveSparkline}
          color="#39d353"
          icon="📈"
        />
        <MetricCard
          title="Total Improvement"
          value={totalImprovement}
          subtitle="base → adaptive"
          trend={metrics.adaptive_last100_mean < metrics.base_mse_last100_mean ? 'up' : 'down'}
          color="#a371f7"
          icon="🎯"
        />
        <MetricCard
          title="Hidden μ"
          value={meta.mu.toFixed(4)}
          subtitle={meta.env_mode === 'linear' ? 'Linear' : 'Nonlinear'}
          color="#a371f7"
          icon="🔮"
        />
        <ProgressCard
          title="Updates Committed"
          current={metrics.updates_committed}
          total={metrics.updates_attempted}
          color="#238636"
        />
      </div>

      {/* Learning Insights */}
      <LearningInsightsCard
        baseMSE={metrics.base_mse_last100_mean}
        sessionStartMSE={metrics.session_no_update_last100_mean}
        adaptiveMSE={metrics.adaptive_last100_mean}
      />

      {/* Primary MSE Chart - Three lines */}
      <MSEComparisonChart perStep={perStep} updateEvents={updateEvents} height={350} />

      {/* Transaction Timeline */}
      <TransactionTimeline events={updateEvents} maxTime={maxTime} />

      {/* Run History Panel */}
      <RunHistoryPanel runs={runs} />

      {/* Session Info Cards */}
      <div className="grid grid-cols-2 gap-4">
        <SessionCard meta={meta} />
        <ConfigCard config={meta.plasticity_cfg} />
      </div>
    </div>
  );
}
