import { motion } from 'framer-motion';
import { useMemo } from 'react';
import {
  LineChart,
  Line,
  ResponsiveContainer,
} from 'recharts';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  sparklineData?: number[];
  color?: string;
  icon?: string;
}

export function MetricCard({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  sparklineData,
  color = '#58a6ff',
  icon,
}: MetricCardProps) {
  const chartData = useMemo(() => {
    if (!sparklineData) return null;
    return sparklineData.map((v, i) => ({ i, v }));
  }, [sparklineData]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="metric-card bg-surface-50 border border-surface-200 rounded-lg p-4 relative overflow-hidden"
    >
      {/* Background sparkline */}
      {chartData && (
        <div className="absolute inset-0 opacity-20">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <Line
                type="monotone"
                dataKey="v"
                stroke={color}
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Content */}
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-text-muted uppercase tracking-wider">{title}</span>
          {icon && <span className="text-lg">{icon}</span>}
        </div>

        <div className="flex items-baseline gap-2">
          <span
            className="text-2xl font-bold font-mono"
            style={{ color }}
          >
            {typeof value === 'number' ? value.toFixed(6) : value}
          </span>

          {trend && trendValue && (
            <span
              className={`text-sm font-medium ${
                trend === 'up'
                  ? 'text-accent-green'
                  : trend === 'down'
                  ? 'text-accent-red'
                  : 'text-text-muted'
              }`}
            >
              {trend === 'up' && '↑'}
              {trend === 'down' && '↓'}
              {trendValue}
            </span>
          )}
        </div>

        {subtitle && (
          <p className="text-xs text-text-muted mt-1">{subtitle}</p>
        )}
      </div>
    </motion.div>
  );
}

// Circular progress variant for commit rate
interface ProgressCardProps {
  title: string;
  current: number;
  total: number;
  color?: string;
}

export function ProgressCard({ title, current, total, color = '#238636' }: ProgressCardProps) {
  const percentage = total > 0 ? (current / total) * 100 : 0;
  const circumference = 2 * Math.PI * 36;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="metric-card bg-surface-50 border border-surface-200 rounded-lg p-4 flex items-center gap-4"
    >
      {/* Circular progress */}
      <div className="relative w-20 h-20">
        <svg className="w-20 h-20 transform -rotate-90">
          <circle
            cx="40"
            cy="40"
            r="36"
            className="stroke-surface-200"
            fill="none"
            strokeWidth="6"
          />
          <motion.circle
            cx="40"
            cy="40"
            r="36"
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold" style={{ color }}>
            {percentage.toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Text */}
      <div>
        <p className="text-xs text-text-muted uppercase tracking-wider">{title}</p>
        <p className="text-xl font-bold font-mono">
          <span style={{ color }}>{current}</span>
          <span className="text-text-muted"> / {total}</span>
        </p>
      </div>
    </motion.div>
  );
}
