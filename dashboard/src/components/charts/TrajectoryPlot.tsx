import { useMemo, useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import type { TrajectoryPoint } from '../../types';

interface TrajectoryPlotProps {
  trajectory: TrajectoryPoint[];
  height?: number;
}

export function TrajectoryPlot({ trajectory, height = 400 }: TrajectoryPlotProps) {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(10);
  const animationRef = useRef<number>();

  // Compute bounds
  const bounds = useMemo(() => {
    const xs = trajectory.map((p) => p.x);
    const ys = trajectory.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padding = 0.1;
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    return {
      minX: minX - rangeX * padding,
      maxX: maxX + rangeX * padding,
      minY: minY - rangeY * padding,
      maxY: maxY + rangeY * padding,
    };
  }, [trajectory]);

  // Convert point to SVG coordinates
  const toSvg = (x: number, y: number, width: number, height: number) => {
    const svgX = ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * width;
    const svgY = height - ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * height;
    return { svgX, svgY };
  };

  // Animation loop
  useEffect(() => {
    if (isPlaying) {
      const animate = () => {
        setCurrentTime((t) => {
          const next = t + playbackSpeed;
          if (next >= trajectory.length - 1) {
            setIsPlaying(false);
            return trajectory.length - 1;
          }
          return next;
        });
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, trajectory.length]);

  // Path data
  const pathData = useMemo(() => {
    const points = trajectory.slice(0, Math.floor(currentTime) + 1);
    if (points.length < 2) return '';

    return points
      .map((p, i) => {
        const { svgX, svgY } = toSvg(p.x, p.y, 500, height - 60);
        return `${i === 0 ? 'M' : 'L'} ${svgX} ${svgY}`;
      })
      .join(' ');
  }, [trajectory, currentTime, bounds, height]);

  const currentPoint = trajectory[Math.floor(currentTime)];
  const currentSvg = currentPoint ? toSvg(currentPoint.x, currentPoint.y, 500, height - 60) : null;

  return (
    <div className="bg-surface-50 border border-surface-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-text-primary">2D Trajectory</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              if (currentTime >= trajectory.length - 1) {
                setCurrentTime(0);
              }
              setIsPlaying(!isPlaying);
            }}
            className="px-3 py-1 bg-surface-100 hover:bg-surface-200 rounded text-sm transition-colors"
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            onClick={() => {
              setIsPlaying(false);
              setCurrentTime(0);
            }}
            className="px-3 py-1 bg-surface-100 hover:bg-surface-200 rounded text-sm transition-colors"
          >
            ⏹ Reset
          </button>
          <select
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
            className="bg-surface-100 border border-surface-200 rounded px-2 py-1 text-sm"
          >
            <option value="1">1×</option>
            <option value="5">5×</option>
            <option value="10">10×</option>
            <option value="20">20×</option>
          </select>
        </div>
      </div>

      {/* SVG Canvas */}
      <svg
        width="100%"
        height={height - 60}
        viewBox={`0 0 500 ${height - 60}`}
        className="bg-surface rounded-lg"
      >
        {/* Grid */}
        <defs>
          <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#21262d" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* Trajectory path */}
        <motion.path
          d={pathData}
          fill="none"
          stroke="url(#trajectoryGradient)"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Gradient for time coloring */}
        <defs>
          <linearGradient id="trajectoryGradient" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#58a6ff" />
            <stop offset="100%" stopColor="#da3633" />
          </linearGradient>
        </defs>

        {/* Current position */}
        {currentSvg && (
          <g>
            {/* Glow effect */}
            <circle
              cx={currentSvg.svgX}
              cy={currentSvg.svgY}
              r="12"
              fill="#39d353"
              opacity="0.3"
            />
            {/* Main dot */}
            <circle
              cx={currentSvg.svgX}
              cy={currentSvg.svgY}
              r="6"
              fill="#39d353"
              stroke="#0d1117"
              strokeWidth="2"
            />

            {/* Velocity vector */}
            {currentPoint && (
              <line
                x1={currentSvg.svgX}
                y1={currentSvg.svgY}
                x2={currentSvg.svgX + currentPoint.vx * 10}
                y2={currentSvg.svgY - currentPoint.vy * 10}
                stroke="#f0883e"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
              />
            )}
          </g>
        )}

        {/* Arrow marker */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#f0883e" />
          </marker>
        </defs>

        {/* Origin marker */}
        <circle
          cx={toSvg(0, 0, 500, height - 60).svgX}
          cy={toSvg(0, 0, 500, height - 60).svgY}
          r="4"
          fill="#6e7681"
          stroke="#8b949e"
          strokeWidth="1"
        />
      </svg>

      {/* Timeline scrubber */}
      <div className="mt-4">
        <input
          type="range"
          min={0}
          max={trajectory.length - 1}
          value={currentTime}
          onChange={(e) => {
            setIsPlaying(false);
            setCurrentTime(Number(e.target.value));
          }}
          className="w-full accent-accent-blue"
        />
        <div className="flex justify-between text-xs text-text-muted mt-1">
          <span>t=0</span>
          <span className="font-mono text-accent-blue">t={Math.floor(currentTime)}</span>
          <span>t={trajectory.length - 1}</span>
        </div>
      </div>

      {/* Current state display */}
      {currentPoint && (
        <div className="mt-4 grid grid-cols-4 gap-2 text-xs">
          <div className="bg-surface-100 rounded px-2 py-1">
            <span className="text-text-muted block">x</span>
            <span className="font-mono text-text-primary">{currentPoint.x.toFixed(3)}</span>
          </div>
          <div className="bg-surface-100 rounded px-2 py-1">
            <span className="text-text-muted block">y</span>
            <span className="font-mono text-text-primary">{currentPoint.y.toFixed(3)}</span>
          </div>
          <div className="bg-surface-100 rounded px-2 py-1">
            <span className="text-text-muted block">vx</span>
            <span className="font-mono text-accent-orange">{currentPoint.vx.toFixed(3)}</span>
          </div>
          <div className="bg-surface-100 rounded px-2 py-1">
            <span className="text-text-muted block">vy</span>
            <span className="font-mono text-accent-orange">{currentPoint.vy.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
