import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { valueToColor } from '../../utils/formatting';

interface WeightHeatmapProps {
  name: string;
  data: number[][];
  description?: string;
}

export function WeightHeatmap({ name, data, description }: WeightHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null);

  // Calculate statistics
  const stats = useMemo(() => {
    const flat = data.flat();
    const sum = flat.reduce((a, b) => a + b, 0);
    const mean = sum / flat.length;
    const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length;
    const std = Math.sqrt(variance);
    const maxAbs = Math.max(...flat.map(Math.abs));
    const frobNorm = Math.sqrt(flat.reduce((a, b) => a + b * b, 0));

    return { mean, std, maxAbs, frobNorm };
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
            {rows} × {cols}
            {description && ` — ${description}`}
          </p>
        </div>
        <div className="text-xs text-text-muted">
          ‖·‖ = {stats.frobNorm.toFixed(3)}
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
                className="heatmap-cell relative"
                style={{
                  backgroundColor: valueToColor(value, stats.maxAbs),
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
            <span className="font-mono text-text-primary ml-2">
              {hoveredCell.value.toFixed(4)}
            </span>
          </div>
        )}
      </div>

      {/* Color scale legend */}
      <div className="mt-3 flex items-center gap-2">
        <span className="text-xs text-text-muted">−{stats.maxAbs.toFixed(2)}</span>
        <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-red-500 via-white to-blue-500" />
        <span className="text-xs text-text-muted">+{stats.maxAbs.toFixed(2)}</span>
      </div>

      {/* Stats row */}
      <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">Mean</span>
          <span className="font-mono text-text-primary">{stats.mean.toFixed(4)}</span>
        </div>
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">Std</span>
          <span className="font-mono text-text-primary">{stats.std.toFixed(4)}</span>
        </div>
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">Max |·|</span>
          <span className="font-mono text-text-primary">{stats.maxAbs.toFixed(4)}</span>
        </div>
        <div className="bg-surface-100 rounded px-2 py-1">
          <span className="text-text-muted block">‖·‖_F</span>
          <span className="font-mono text-text-primary">{stats.frobNorm.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}

// Compact version for overview
export function WeightHeatmapMini({ name, data }: { name: string; data: number[][] }) {
  const flat = data.flat();
  const maxAbs = Math.max(...flat.map(Math.abs));

  const rows = data.length;
  const cols = data[0]?.length ?? 0;
  const cellSize = Math.min(4, 100 / Math.max(rows, cols));

  return (
    <div className="bg-surface-100 rounded p-2">
      <p className="text-xs font-mono text-text-muted mb-1">{name}</p>
      <div
        className="grid gap-px"
        style={{
          gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
        }}
      >
        {data.map((row, i) =>
          row.map((value, j) => (
            <div
              key={`${i}-${j}`}
              style={{
                backgroundColor: valueToColor(value, maxAbs),
                width: cellSize,
                height: cellSize,
              }}
            />
          ))
        )}
      </div>
    </div>
  );
}
