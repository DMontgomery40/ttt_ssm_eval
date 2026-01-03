import { useDashboardStore } from '../../store';
import { MetricCard, ProgressCard } from '../cards/MetricCard';
import { SessionCard, ConfigCard } from '../cards/SessionCard';
import { MSEComparisonChart } from '../charts/MSEComparisonChart';
import { TransactionTimeline } from '../charts/TransactionTimeline';
import { formatImprovement, formatRelativeTime } from '../../utils/formatting';

export function OverviewTab() {
  const { currentSession } = useDashboardStore();
  const { meta, metrics, perStep, updateEvents } = currentSession;

  // Sparkline data (last 100 points)
  const baselineSparkline = perStep.slice(-100).map((s) => s.baseline_mse);
  const adaptiveSparkline = perStep.slice(-100).map((s) => s.adaptive_mse);

  const improvement = formatImprovement(
    metrics.baseline_mse_last100_mean,
    metrics.adaptive_mse_last100_mean
  );

  const maxTime = perStep.length > 0 ? perStep[perStep.length - 1].t : 0;

  return (
    <div className="space-y-6">
      {/* Hero Metrics Row */}
      <div className="grid grid-cols-6 gap-4">
        <MetricCard
          title="Baseline MSE (last 100)"
          value={metrics.baseline_mse_last100_mean}
          sparklineData={baselineSparkline}
          color="#6e7681"
          icon="📉"
        />
        <MetricCard
          title="Adaptive MSE (last 100)"
          value={metrics.adaptive_mse_last100_mean}
          sparklineData={adaptiveSparkline}
          color="#39d353"
          icon="📈"
        />
        <MetricCard
          title="Improvement"
          value={improvement}
          subtitle="(baseline - adaptive) / baseline"
          trend={metrics.adaptive_mse_last100_mean < metrics.baseline_mse_last100_mean ? 'up' : 'down'}
          color="#238636"
          icon="🎯"
        />
        <ProgressCard
          title="Updates Committed"
          current={metrics.updates_committed}
          total={metrics.updates_attempted}
          color="#238636"
        />
        <MetricCard
          title="Hidden μ"
          value={meta.mu.toFixed(4)}
          subtitle={`Range: [0.02, 0.25]`}
          color="#a371f7"
          icon="🔮"
        />
        <MetricCard
          title="Session Age"
          value={formatRelativeTime(meta.created_at_unix)}
          subtitle={meta.env_mode === 'linear' ? 'Linear friction' : 'Nonlinear friction'}
          color="#58a6ff"
          icon="⏱️"
        />
      </div>

      {/* Primary MSE Chart */}
      <MSEComparisonChart perStep={perStep} updateEvents={updateEvents} height={350} />

      {/* Transaction Timeline */}
      <TransactionTimeline events={updateEvents} maxTime={maxTime} />

      {/* Session Info Cards */}
      <div className="grid grid-cols-2 gap-4">
        <SessionCard meta={meta} />
        <ConfigCard config={meta.plasticity_cfg} />
      </div>
    </div>
  );
}
