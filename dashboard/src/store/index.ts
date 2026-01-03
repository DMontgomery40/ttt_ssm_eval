import { create } from 'zustand';
import type { SessionData, TabId, UpdateEvent, SessionIndex, SessionTreeNode, RunData } from '../types';
import { mockSession, mockSessions, mockSessionIndex, mockSessionTree, getSessionLineage } from '../data/mockData';

interface DashboardState {
  // Current session
  currentSession: SessionData;
  sessions: SessionData[];

  // Phase 1: Session index and tree
  sessionIndex: SessionIndex;
  sessionTree: SessionTreeNode[];

  // Navigation
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;

  // Time control
  currentTime: number;
  isPlaying: boolean;
  playbackSpeed: number;
  setCurrentTime: (t: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;

  // Selection
  selectedUpdateEvent: UpdateEvent | null;
  setSelectedUpdateEvent: (event: UpdateEvent | null) => void;

  // Phase 1: Run selection
  selectedRunId: string | null;
  setSelectedRunId: (runId: string | null) => void;
  getCurrentRun: () => RunData | null;

  // View options
  logScale: boolean;
  smoothing: boolean;
  smoothingWindow: number;
  setLogScale: (log: boolean) => void;
  setSmoothing: (smooth: boolean) => void;
  setSmoothingWindow: (window: number) => void;

  // Phase 1: Weight comparison mode
  weightCompareMode: 'none' | 'parent' | 'base';
  setWeightCompareMode: (mode: 'none' | 'parent' | 'base') => void;

  // Session management
  selectedSessionIds: string[];
  toggleSessionSelection: (id: string) => void;
  setCurrentSession: (session: SessionData) => void;

  // Phase 1: Session lineage
  getSessionLineage: () => string[];

  // Phase 1: Fork session (mock implementation)
  forkSession: (parentSessionId: string, newSessionId: string) => void;
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  // Initial state
  currentSession: mockSession,
  sessions: mockSessions,
  sessionIndex: mockSessionIndex,
  sessionTree: mockSessionTree,
  activeTab: 'session-tree',  // Phase 1: Start on session tree
  currentTime: 0,
  isPlaying: false,
  playbackSpeed: 1,
  selectedUpdateEvent: null,
  selectedRunId: null,
  logScale: false,
  smoothing: false,
  smoothingWindow: 10,
  weightCompareMode: 'none',
  selectedSessionIds: [],

  // Actions
  setActiveTab: (tab) => set({ activeTab: tab }),

  setCurrentTime: (t) => set({ currentTime: t }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  setSelectedUpdateEvent: (event) => set({ selectedUpdateEvent: event }),

  setSelectedRunId: (runId) => set({ selectedRunId: runId }),

  getCurrentRun: () => {
    const state = get();
    const runId = state.selectedRunId;
    if (!runId) return state.currentSession.runs[state.currentSession.runs.length - 1] || null;
    return state.currentSession.runs.find(r => r.run_id === runId) || null;
  },

  setLogScale: (log) => set({ logScale: log }),
  setSmoothing: (smooth) => set({ smoothing: smooth }),
  setSmoothingWindow: (window) => set({ smoothingWindow: window }),

  setWeightCompareMode: (mode) => set({ weightCompareMode: mode }),

  toggleSessionSelection: (id) =>
    set((state) => ({
      selectedSessionIds: state.selectedSessionIds.includes(id)
        ? state.selectedSessionIds.filter((i) => i !== id)
        : [...state.selectedSessionIds, id]
    })),

  setCurrentSession: (session) => set({
    currentSession: session,
    selectedRunId: null,  // Reset run selection when switching sessions
    selectedUpdateEvent: null
  }),

  getSessionLineage: () => {
    const state = get();
    return getSessionLineage(state.currentSession.meta.session_id, state.sessionIndex);
  },

  forkSession: (parentSessionId, newSessionId) => {
    // In a real implementation, this would create a new session on the backend
    // For now, we just add it to the mock index
    set((state) => {
      const parentSession = state.sessionIndex.sessions[parentSessionId];
      if (!parentSession) return state;

      const newSummary = {
        session_id: newSessionId,
        parent_session_id: parentSessionId,
        root_session_id: parentSession.root_session_id,
        created_at_unix: Math.floor(Date.now() / 1000),
        last_run_at_unix: null,
        env_mode: parentSession.env_mode,
        mu: parentSession.mu,
        model_signature: parentSession.model_signature,
        total_runs: 0,
        total_updates_committed: 0,
        total_updates_rolled_back: 0
      };

      const newSessions = {
        ...state.sessionIndex.sessions,
        [newSessionId]: newSummary
      };

      return {
        sessionIndex: {
          ...state.sessionIndex,
          sessions: newSessions
        }
      };
    });
  }
}));
