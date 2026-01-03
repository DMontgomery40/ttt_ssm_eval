import { create } from 'zustand';
import type { SessionData, TabId, UpdateEvent } from '../types';
import { mockSession, mockSessions } from '../data/mockData';

interface DashboardState {
  // Current session
  currentSession: SessionData;
  sessions: SessionData[];

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

  // View options
  logScale: boolean;
  smoothing: boolean;
  smoothingWindow: number;
  setLogScale: (log: boolean) => void;
  setSmoothing: (smooth: boolean) => void;
  setSmoothingWindow: (window: number) => void;

  // Session management
  selectedSessionIds: string[];
  toggleSessionSelection: (id: string) => void;
  setCurrentSession: (session: SessionData) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  // Initial state
  currentSession: mockSession,
  sessions: mockSessions,
  activeTab: 'overview',
  currentTime: 0,
  isPlaying: false,
  playbackSpeed: 1,
  selectedUpdateEvent: null,
  logScale: false,
  smoothing: false,
  smoothingWindow: 10,
  selectedSessionIds: [],

  // Actions
  setActiveTab: (tab) => set({ activeTab: tab }),

  setCurrentTime: (t) => set({ currentTime: t }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  setSelectedUpdateEvent: (event) => set({ selectedUpdateEvent: event }),

  setLogScale: (log) => set({ logScale: log }),
  setSmoothing: (smooth) => set({ smoothing: smooth }),
  setSmoothingWindow: (window) => set({ smoothingWindow: window }),

  toggleSessionSelection: (id) =>
    set((state) => ({
      selectedSessionIds: state.selectedSessionIds.includes(id)
        ? state.selectedSessionIds.filter((i) => i !== id)
        : [...state.selectedSessionIds, id]
    })),

  setCurrentSession: (session) => set({ currentSession: session })
}));
