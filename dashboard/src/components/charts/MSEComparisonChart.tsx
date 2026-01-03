import { useMemo, useCallback } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
  Scatter,
} from 'recharts';
import { useDashboardStore } from '../../store';
import { smoothData, formatNumber, getStatusColor } from '../../utils/formatting';
import type { PerStepMetric, UpdateEvent } from '../../types';

interface MSEComparisonChartProps {
  perStep: PerStepMetric[];
  updateEvents: UpdateEvent[];
  height?: number;
}

export function MSEComparisonChart({
  perStep,
  updateEvents,
  height = 350,
}: MSEComparisonChartProps) {
  const { logScale, smoothing, smoothingWindow, setSelectedUpdateEvent } = useDashboardStore();

  // Prepare data
  const chartData = useMemo(() => {
    let baselineValues = perStep.map((s) => s.baseline_mse);
    let adaptiveValues = perStep.map((s) => s.adaptive_mse);

    if (smoothing) {
      baselineValues = smoothData(baselineValues, smoothingWindow);
      adaptiveValues = smoothData(adaptiveValues, smoothingWindow);
    }

    return perStep.map((step, i) => ({
      t: step.t,
      baseline: baselineValues[i],
      adaptive: adaptiveValues[i],
      gap: baselineValues[i] - adaptiveValues[i],
      isUpdate: step.did_update,
      updateOk: step.update_ok,
    }));
  }, [perStep, smoothing, smoothingWindow]);

  // Update event markers
  const updateMarkers = useMemo(() => {
    return updateEvents.map((event) => ({
      t: event.t,
      y: chartData.find((d) => d.t === event.t)?.adaptive ?? 0,
      status: event.status,
      event,
    }));
  }, [updateEvents, chartData]);

  const handleMarkerClick = useCallback((event: UpdateEvent) => {
    setSelectedUpdateEvent(event);
  }, [setSelectedUpdateEvent]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0]?.payload;
    const updateEvent = updateEvents.find((e) => e.t === label);

    return (
      <div className="bg-surface-50 border border-surface-200 rounded-lg p-3 shadow-lg">
        <p className="text-xs text-text-muted mb-2">Timestep {label}</p>
        <div className="space-y-1 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-text-muted" />
            <span className="text-text-secondary">Baseline:</span>
            <span className="font-mono text-text-primary">{formatNumber(data?.baseline)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-accent-cyan" />
            <span className="text-text-secondary">Adaptive:</span>
            <span className="font-mono text-accent-cyan">{formatNumber(data?.adaptive)}</span>
          </div>
          {data?.gap > 0 && (
            <div className="text-xs text-accent-green mt-1">
              Learning gap: {formatNumber(data.gap)}
            </div>
          )}
        </div>
        {updateEvent && (
          <div
            className="mt-2 pt-2 border-t border-surface-200 text-xs"
            style={{ color: getStatusColor(updateEvent.status) }}
          >
            Update: {updateEvent.status}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-text-primary">MSE Comparison</h3>
        <div className="flex items-center gap-4 text-xs">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={logScale}
              onChange={(e) => useDashboardStore.getState().setLogScale(e.target.checked)}
              className="rounded border-surface-200 bg-surface-100"
            />
            <span className="text-text-secondary">Log scale</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={smoothing}
              onChange={(e) => useDashboardStore.getState().setSmoothing(e.target.checked)}
              className="rounded border-surface-200 bg-surface-100"
            />
            <span className="text-text-secondary">Smooth</span>
          </label>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
          <defs>
            <linearGradient id="gapGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#39d353" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#39d353" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />

          <XAxis
            dataKey="t"
            stroke="#6e7681"
            tick={{ fill: '#8b949e', fontSize: 11 }}
            tickLine={{ stroke: '#30363d' }}
          />

          <YAxis
            scale={logScale ? 'log' : 'auto'}
            domain={logScale ? ['auto', 'auto'] : [0, 'auto']}
            stroke="#6e7681"
            tick={{ fill: '#8b949e', fontSize: 11 }}
            tickLine={{ stroke: '#30363d' }}
            tickFormatter={(v) => v.toExponential(1)}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Learning gap area */}
          <Area
            type="monotone"
            dataKey="gap"
            fill="url(#gapGradient)"
            stroke="none"
          />

          {/* Baseline line */}
          <Line
            type="monotone"
            dataKey="baseline"
            stroke="#6e7681"
            strokeWidth={2}
            dot={false}
            name="Baseline"
          />

          {/* Adaptive line with glow effect */}
          <Line
            type="monotone"
            dataKey="adaptive"
            stroke="#39d353"
            strokeWidth={2.5}
            dot={false}
            name="Adaptive"
            style={{ filter: 'drop-shadow(0 0 4px rgba(57, 211, 83, 0.5))' }}
          />

          {/* Update event markers */}
          {updateMarkers.map((marker, i) => (
            <ReferenceLine
              key={i}
              x={marker.t}
              stroke={getStatusColor(marker.status)}
              strokeDasharray="4 4"
              strokeWidth={1}
              strokeOpacity={0.5}
            />
          ))}

          {/* Scatter for update markers */}
          <Scatter
            data={updateMarkers}
            dataKey="y"
            fill="#fff"
            shape={(props: any) => {
              const { cx, cy, payload } = props;
              const color = getStatusColor(payload.status);
              const isCommit = payload.status === 'commit';

              return (
                <g
                  onClick={() => handleMarkerClick(payload.event)}
                  style={{ cursor: 'pointer' }}
                >
                  {isCommit ? (
                    <polygon
                      points={`${cx},${cy - 6} ${cx + 5},${cy + 4} ${cx - 5},${cy + 4}`}
                      fill={color}
                      stroke="#0d1117"
                      strokeWidth={1}
                    />
                  ) : (
                    <g>
                      <line
                        x1={cx - 4}
                        y1={cy - 4}
                        x2={cx + 4}
                        y2={cy + 4}
                        stroke={color}
                        strokeWidth={2}
                      />
                      <line
                        x1={cx + 4}
                        y1={cy - 4}
                        x2={cx - 4}
                        y2={cy + 4}
                        stroke={color}
                        strokeWidth={2}
                      />
                    </g>
                  )}
                </g>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-text-muted" />
          <span className="text-text-muted">Baseline (frozen)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-accent-cyan" style={{ boxShadow: '0 0 4px rgba(57, 211, 83, 0.5)' }} />
          <span className="text-text-secondary">Adaptive (TTT)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-0 h-0 border-l-4 border-r-4 border-b-6 border-transparent border-b-accent-green" />
          <span className="text-text-secondary">Committed</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-accent-red font-bold">✕</span>
          <span className="text-text-secondary">Rolled back</span>
        </div>
      </div>
    </div>
  );
}
