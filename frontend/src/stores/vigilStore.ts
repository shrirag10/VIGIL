import { create } from 'zustand'
import type { Camera, SystemStats, SystemParams, Violation, DetectionEvent, ViolationStats, TabId, AudioSettings, Recording, Report, TamperStatus } from '../types'
import * as api from '../lib/api'

interface VigilState {
  // Navigation
  activeTab: TabId
  setActiveTab: (tab: TabId) => void

  // Cameras
  cameras: Camera[]
  fetchCameras: () => Promise<void>

  // System
  stats: SystemStats | null
  params: SystemParams | null
  fetchStats: () => Promise<void>
  fetchParams: () => Promise<void>

  // Events & Violations
  events: DetectionEvent[]
  violations: Violation[]
  violationStats: ViolationStats | null
  fetchEvents: () => Promise<void>
  fetchViolations: (cameraId?: number) => Promise<void>
  fetchViolationStats: () => Promise<void>

  // Audio
  audioSettings: AudioSettings | null
  fetchAudioSettings: () => Promise<void>

  // Tamper
  tamperStatus: Record<number, TamperStatus>
  fetchTamperStatus: () => Promise<void>

  // Recordings
  recordings: Recording[]
  fetchRecordings: () => Promise<void>

  // Reports
  reports: Report[]
  fetchReports: () => Promise<void>

  // Polling
  startPolling: () => void
  stopPolling: () => void
}

let pollIntervals: ReturnType<typeof setInterval>[] = []

export const useVigilStore = create<VigilState>((set, get) => ({
  activeTab: 'dashboard',
  setActiveTab: (tab) => set({ activeTab: tab }),

  cameras: [],
  fetchCameras: async () => {
    try {
      const data = await api.getCameras()
      set({ cameras: data.cameras })
    } catch { /* ignore */ }
  },

  stats: null,
  params: null,
  fetchStats: async () => {
    try {
      const data = await api.getSystemStats()
      set({ stats: data })
    } catch { /* ignore */ }
  },
  fetchParams: async () => {
    try {
      const data = await api.getSystemParams()
      set({ params: data })
    } catch { /* ignore */ }
  },

  events: [],
  violations: [],
  violationStats: null,
  fetchEvents: async () => {
    try {
      const data = await api.getEvents()
      set({ events: data.events })
    } catch { /* ignore */ }
  },
  fetchViolations: async (cameraId) => {
    try {
      const data = await api.getViolations(cameraId, 100)
      set({ violations: data.violations })
    } catch { /* ignore */ }
  },
  fetchViolationStats: async () => {
    try {
      const data = await api.getViolationStats()
      set({ violationStats: data })
    } catch { /* ignore */ }
  },

  audioSettings: null,
  fetchAudioSettings: async () => {
    try {
      const data = await api.getAudioSettings()
      set({ audioSettings: data })
    } catch { /* ignore */ }
  },

  tamperStatus: {},
  fetchTamperStatus: async () => {
    try {
      const data = await api.getTamperStatus()
      set({ tamperStatus: data.tamper_status })
    } catch { /* ignore */ }
  },

  recordings: [],
  fetchRecordings: async () => {
    try {
      const data = await api.getRecordings()
      set({ recordings: data.recordings })
    } catch { /* ignore */ }
  },

  reports: [],
  fetchReports: async () => {
    try {
      const data = await api.getReports()
      set({ reports: data.reports })
    } catch { /* ignore */ }
  },

  startPolling: () => {
    // Clear any existing intervals first (prevents doubling on HMR / re-mount)
    pollIntervals.forEach(clearInterval)
    pollIntervals = []

    const { fetchCameras, fetchStats, fetchParams, fetchEvents, fetchViolations, fetchTamperStatus } = get()

    // Initial fetch
    fetchCameras()
    fetchStats()
    fetchParams()
    fetchEvents()
    fetchViolations()
    fetchTamperStatus()

    // Polling intervals
    pollIntervals = [
      setInterval(fetchCameras, 2000),
      setInterval(fetchStats, 2000),
      setInterval(fetchParams, 3000),
      setInterval(fetchEvents, 3000),
      setInterval(fetchViolations, 5000),
      setInterval(fetchTamperStatus, 5000),
    ]
  },

  stopPolling: () => {
    pollIntervals.forEach(clearInterval)
    pollIntervals = []
  },
}))
