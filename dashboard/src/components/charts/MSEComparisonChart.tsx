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

  // Prepare data with three lines
  const chartData = useMemo(() => {
    let baseValues = perStep.map((s) => s.base_mse);
    let sessionStartValues = perStep.map((s) => s.session_start_mse);
    let adaptiveValues = perStep.map((s) => s.adaptive_mse);

    if (smoothing) {
      baseValues = smoothData(baseValues, smoothingWindow);
      sessionStartValues = smoothData(sessionStartValues, smoothingWindow);
      adaptiveValues = smoothData(adaptiveValues, smoothingWindow);
    }

    return perStep.map((step, i) => ({
      t: step.t,
      base: baseValues[i],
      sessionStart: sessionStartValues[i],
      adaptive: adaptiveValues[i],
      // Gap between session start and adaptive (online learning region)
      onlineGap: Math.max(0, sessionStartValues[i] - adaptiveValues[i]),
      // Gap between base and session start (persistent learning region)
      persistentGap: Math.max(0, baseValues[i] - sessionStartValues[i]),
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
            <span className="text-text-secondary">Base:</span>
            <span className="font-mono text-text-primary">{formatNumber(data?.base)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-accent-blue" />
            <span className="text-text-secondary">Session Start:</span>
            <span className="font-mono text-accent-blue">{formatNumber(data?.sessionStart)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-accent-green" />
            <span className="text-text-secondary">Adaptive:</span>
            <span className="font-mono text-accent-green">{formatNumber(data?.adaptive)}</span>
          </div>
          {data?.persistentGap > 0.0001 && (
            <div className="text-xs text-accent-blue mt-1">
              Persistent gap: {formatNumber(data.persistentGap)}
            </div>
          )}
          {data?.onlineGap > 0.0001 && (
            <div className="text-xs text-accent-green mt-1">
              Online gap: {formatNumber(data.onlineGap)}
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
        <h3 className="text-sm font-semibold text-text-primary">MSE Comparison (Three-Way)</h3>
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
            {/* Persistent learning region fill (base to session start) */}
            <linearGradient id="persistentGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
            </linearGradient>
            {/* Online learning region fill (session start to adaptive) */}
            <linearGradient id="onlineGradient" x1="0" y1="0" x2="0" y2="1">
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

          {/* Online learning gap area (session start to adaptive) */}
          <Area
            type="monotone"
            dataKey="onlineGap"
            fill="url(#onlineGradient)"
            stroke="none"
          />

          {/* Base line - gray, dashed, thin */}
          <Line
            type="monotone"
            dataKey="base"
            stroke="#6e7681"
            strokeWidth={1.5}
            strokeDasharray="5 5"
            dot={false}
            name="Base"
          />

          {/* Session start line - blue, dashed, medium */}
          <Line
            type="monotone"
            dataKey="sessionStart"
            stroke="#58a6ff"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Session Start"
          />

          {/* Adaptive line - green, solid, thick with glow */}
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
          <div className="w-4 h-0.5 bg-text-muted" style={{ borderTop: '1.5px dashed #6e7681' }} />
          <span className="text-text-muted">Base (frozen)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-accent-blue" style={{ borderTop: '2px dashed #58a6ff' }} />
          <span className="text-text-secondary">Session Start</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-accent-green rounded" style={{ boxShadow: '0 0 4px rgba(57, 211, 83, 0.5)' }} />
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
