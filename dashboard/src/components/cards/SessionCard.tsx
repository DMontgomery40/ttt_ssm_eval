import { motion } from 'framer-motion';
import type { SessionMeta, PlasticityConfig } from '../../types';
import { formatRelativeTime, truncateHash } from '../../utils/formatting';

interface SessionCardProps {
  meta: SessionMeta;
}

export function SessionCard({ meta }: SessionCardProps) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-surface-50 border border-surface-200 rounded-lg p-4"
    >
      <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
        <span>📋</span> Session Identity
      </h3>

      <div className="space-y-3 text-sm">
        {/* Session ID */}
        <div className="flex justify-between">
          <span className="text-text-muted">Session ID</span>
          <span className="font-mono text-accent-blue">{meta.session_id}</span>
        </div>

        {/* Environment */}
        <div className="flex justify-between">
          <span className="text-text-muted">Environment</span>
          <span className={`font-medium ${meta.env_mode === 'linear' ? 'text-accent-green' : 'text-accent-purple'}`}>
            {meta.env_mode}
          </span>
        </div>

        {/* Hidden μ */}
        <div className="flex justify-between">
          <span className="text-text-muted">Hidden μ</span>
          <span className="font-mono font-bold text-accent-cyan">{meta.mu.toFixed(4)}</span>
        </div>

        {/* Created */}
        <div className="flex justify-between">
          <span className="text-text-muted">Created</span>
          <span className="text-text-secondary">{formatRelativeTime(meta.created_at_unix)}</span>
        </div>

        <div className="border-t border-surface-200 pt-3 mt-3">
          {/* Model signature */}
          <div className="flex justify-between items-center">
            <span className="text-text-muted">Signature</span>
            <button
              onClick={() => copyToClipboard(meta.model_signature)}
              className="font-mono text-xs text-text-secondary hover:text-accent-blue transition-colors"
              title="Click to copy"
            >
              {truncateHash(meta.model_signature, 6)} 📋
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

interface ConfigCardProps {
  config: PlasticityConfig;
}

export function ConfigCard({ config }: ConfigCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-surface-50 border border-surface-200 rounded-lg p-4"
    >
      <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
        <span>⚙️</span> Plasticity Config
      </h3>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <ConfigItem label="Learning Rate" value={config.lr.toExponential(1)} />
        <ConfigItem label="Momentum" value={config.momentum.toString()} />
        <ConfigItem label="Chunk Size" value={config.chunk.toString()} />
        <ConfigItem label="Buffer Len" value={config.buffer_len.toString()} />
        <ConfigItem label="Rollback Tol" value={`${(config.rollback_tol * 100).toFixed(0)}%`} />
        <ConfigItem label="Grad Max" value={config.grad_norm_max.toString()} />
        <ConfigItem label="NS Steps" value={config.ns_steps.toString()} />
        <ConfigItem label="Nesterov" value={config.nesterov ? 'Yes' : 'No'} />
      </div>
    </motion.div>
  );
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-100 rounded px-2 py-1.5">
      <span className="text-text-muted block">{label}</span>
      <span className="font-mono text-accent-blue">{value}</span>
    </div>
  );
}
