import { useDashboardStore } from '../../store';
import type { TabId } from '../../types';

const tabs: { id: TabId; label: string; icon: string }[] = [
  { id: 'overview', label: 'Overview', icon: '📊' },
  { id: 'weights', label: 'Weights', icon: '🧠' },
  { id: 'transactions', label: 'Transactions', icon: '📝' },
  { id: 'environment', label: 'Physics', icon: '🎯' },
  { id: 'architecture', label: 'Architecture', icon: '🏗️' },
  { id: 'sessions', label: 'Sessions', icon: '📁' },
];

export function Header() {
  const { activeTab, setActiveTab, currentSession } = useDashboardStore();

  return (
    <header className="bg-surface-50 border-b border-surface-200">
      <div className="flex items-center justify-between px-6 py-4">
        {/* Brand */}
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              <span className="text-accent-blue">TTT</span>
              <span className="text-text-secondary">/</span>
              <span className="text-accent-green">SSM</span>
            </h1>
            <p className="text-xs text-text-muted font-mono">Phase 0 Dashboard</p>
          </div>
          <div className="h-8 w-px bg-surface-200" />
          <div className="text-sm">
            <span className="text-text-muted">Session:</span>{' '}
            <span className="font-mono text-text-primary">{currentSession.meta.session_id}</span>
          </div>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-green animate-pulse" />
          <span className="text-sm text-text-secondary">
            μ = {currentSession.meta.mu.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Tab navigation */}
      <nav className="flex px-4 gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg
              transition-all duration-150
              ${
                activeTab === tab.id
                  ? 'bg-surface text-text-primary border-t border-l border-r border-surface-200'
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface-100'
              }
            `}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>
    </header>
  );
}
