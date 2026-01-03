import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';

// Color scale for differences: red (negative) -> white (zero) -> blue (positive)
function diffToColor(value: number, maxAbs: number): string {
  const normalized = Math.max(-1, Math.min(1, value / maxAbs));

  if (normalized >= 0) {
    // Blue for positive (weight increased)
    const intensity = Math.round(255 * (1 - normalized));
    return `rgb(${intensity}, ${intensity}, 255)`;
  } else {
    // Red for negative (weight decreased)
    const intensity = Math.round(255 * (1 + normalized));
    return `rgb(255, ${intensity}, ${intensity})`;
  }
}

interface WeightDiffHeatmapProps {
  name: string;
  data: number[][];
  description?: string;
}

export function WeightDiffHeatmap({ name, data, description }: WeightDiffHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null);

  // Calculate statistics
  const stats = useMemo(() => {
    const flat = data.flat();
    const sum = flat.reduce((a, b) => a + b, 0);
    const mean = sum / flat.length;
    const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length;
    const std = Math.sqrt(variance);
    const maxAbs = Math.max(...flat.map(Math.abs));
    const l2Norm = Math.sqrt(flat.reduce((a, b) => a + b * b, 0));
    const positive = flat.filter(v => v > 0).length;
    const negative = flat.filter(v => v < 0).length;

    return { mean, std, maxAbs, l2Norm, positive, negative, total: flat.length };
  }, [data]);

  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  // Determine cell size based on matrix dimensions
  const cellSize = Math.min(20, Math.max(8, 400 / Math.max(rows, cols)));

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-text-primary font-mono">{name}</h3>
          <p className="text-xs text-text-muted">
            {rows} x {cols}
            {description && ` — ${description}`}
          </p>
        </div>
        <div className="text-xs text-text-muted">
          L2 = {stats.l2Norm.toFixed(4)}
        </div>
      </div>

      {/* Heatmap grid */}
      <div className="relative overflow-auto">
        <div
          className="grid gap-px bg-surface-200 rounded overflow-hidden"
          style={{
            gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
            width: 'fit-content',
          }}
        >
          {data.map((row, i) =>
            row.map((value, j) => (
              <motion.div
                key={`${i}-${j}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: (i * cols + j) * 0.001 }}
                className="relative"
                style={{
                  backgroundColor: diffToColor(value, stats.maxAbs),
                  width: cellSize,
                  height: cellSize,
                }}
                onMouseEnter={() => setHoveredCell({ row: i, col: j, value })}
                onMouseLeave={() => setHoveredCell(null)}
              />
            ))
          )}
        </div>

        {/* Hover tooltip */}
        {hoveredCell && (
          <div className="absolute top-0 right-0 bg-surface-100 border border-surface-200 rounded px-2 py-1 text-xs z-20">
            <span className="text-text-muted">
              [{hoveredCell.row}, {hoveredCell.col}]
            </span>
            <span
              className={`font-mono ml-2 ${
                hoveredCell.value > 0 ? 'text-accent-blue' : hoveredCell.value < 0 ? 'text-accent-red' : 'text-text-muted'
              }`}
            >
              {hoveredCell.value >= 0 ? '+' : ''}{hoveredCell.value.toFixed(4)}
            </span>
          </div>
        )}
      </div>

      {/* Color scale legend */}
      <div className="mt-3 flex items-center gap-2">
        <span className="text-xs text-accent-red">-{stats.maxAbs.toFixed(3)}</span>
        <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-red-500 via-white to-blue-500" />
        <span className="text-xs text-accent-blue">+{stats.maxAbs.toFixed(3)}</span>
      </div>

      {/* Stats row */}
      <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">Mean Δ</span>
          <span className={`font-mono ${stats.mean >= 0 ? 'text-accent-blue' : 'text-accent-red'}`}>
            {stats.mean >= 0 ? '+' : ''}{stats.mean.toFixed(4)}
          </span>
        </div>
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">Std Δ</span>
          <span className="font-mono text-text-primary">{stats.std.toFixed(4)}</span>
        </div>
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">Increased</span>
          <span className="font-mono text-accent-blue">
            {stats.positive} ({((stats.positive / stats.total) * 100).toFixed(0)}%)
          </span>
        </div>
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">Decreased</span>
          <span className="font-mono text-accent-red">
            {stats.negative} ({((stats.negative / stats.total) * 100).toFixed(0)}%)
          </span>
        </div>
      </div>
    </div>
  );
}
