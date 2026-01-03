// Formatting utilities for TTT-SSM Dashboard

export function formatNumber(n: number, decimals = 4): string {
  if (Math.abs(n) < 0.0001 && n !== 0) {
    return n.toExponential(2);
  }
  return n.toFixed(decimals);
}

export function formatPercent(n: number, decimals = 1): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

export function formatImprovement(baseline: number, adaptive: number): string {
  if (baseline === 0) return 'N/A';
  const improvement = ((baseline - adaptive) / baseline) * 100;
  const sign = improvement >= 0 ? '+' : '';
  return `${sign}${improvement.toFixed(1)}%`;
}

export function formatTimestamp(unixSeconds: number): string {
  const date = new Date(unixSeconds * 1000);
  return date.toLocaleString();
}

export function formatRelativeTime(unixSeconds: number): string {
  const now = Math.floor(Date.now() / 1000);
  const diff = now - unixSeconds;

  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function truncateHash(hash: string, length = 8): string {
  if (hash.length <= length * 2) return hash;
  return `${hash.slice(0, length)}...${hash.slice(-length)}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Apply rolling average smoothing
export function smoothData(data: number[], windowSize: number): number[] {
  if (windowSize <= 1) return data;

  const result: number[] = [];
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(data.length, i + Math.ceil(windowSize / 2));
    const window = data.slice(start, end);
    result.push(window.reduce((a, b) => a + b, 0) / window.length);
  }
  return result;
}

// Color utilities
export function getStatusColor(status: string): string {
  switch (status) {
    case 'commit':
      return '#238636';
    case 'rollback_loss_regression':
      return '#da3633';
    case 'rollback_grad_norm':
      return '#d29922';
    case 'rollback_state_norm':
      return '#a371f7';
    default:
      return '#8b949e';
  }
}

export function getStatusLabel(status: string): string {
  switch (status) {
    case 'commit':
      return 'Committed';
    case 'rollback_loss_regression':
      return 'Loss Regression';
    case 'rollback_grad_norm':
      return 'Grad Explosion';
    case 'rollback_state_norm':
      return 'State Explosion';
    default:
      return status;
  }
}

// For diverging color scales (weight heatmaps)
export function valueToColor(value: number, maxAbs: number): string {
  const normalized = Math.max(-1, Math.min(1, value / maxAbs));

  if (normalized >= 0) {
    // Blue to white for positive
    const intensity = Math.round(255 * (1 - normalized));
    return `rgb(${intensity}, ${intensity}, 255)`;
  } else {
    // Red to white for negative
    const intensity = Math.round(255 * (1 + normalized));
    return `rgb(255, ${intensity}, ${intensity})`;
  }
}
