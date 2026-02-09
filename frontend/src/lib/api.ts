const BASE = ''  // proxy handles /api in dev

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`)
  return res.json()
}

// ── Cameras ──
export const getCameras = () => request<{ cameras: import('../types').Camera[] }>('/api/cameras')
export const getCameraStream = (id: number) => `${BASE}/api/camera/${id}/stream`
export const toggleDetection = (id: number, enabled: boolean) =>
  request(`/api/camera/${id}/detection`, { method: 'POST', body: JSON.stringify({ enabled }) })
export const toggleRecording = (id: number, recording: boolean) =>
  request(`/api/camera/${id}/recording`, { method: 'POST', body: JSON.stringify({ recording }) })
export const getCameraHistory = (id: number) =>
  request<{ history: { timestamp: string; count: number }[] }>(`/api/camera/${id}/history`)
export const getCameraDirections = (id: number) =>
  request<{ directions: Record<string, number> }>(`/api/camera/${id}/directions`)

// ── Zones ──
export const getCameraZones = (id: number) =>
  request<{ zones: import('../types').Zone[] }>(`/api/camera/${id}/zones`)
export const addCameraZone = (id: number, zone: Record<string, unknown>) =>
  request(`/api/camera/${id}/zones`, { method: 'POST', body: JSON.stringify(zone) })
export const deleteCameraZone = (cameraId: number, zoneId: string) =>
  request(`/api/camera/${cameraId}/zones/${zoneId}`, { method: 'DELETE' })

// ── System ──
export const getSystemStats = () => request<import('../types').SystemStats>('/api/system/stats')
export const getSystemParams = () => request<import('../types').SystemParams>('/api/system/params')
export const toggleSystemPower = (enabled: boolean) =>
  request('/api/system/power', { method: 'POST', body: JSON.stringify({ enabled }) })
export const shutdownServer = () =>
  request('/api/system/shutdown', { method: 'POST', body: JSON.stringify({}) })

// ── Events & Violations ──
export const getEvents = () => request<{ events: import('../types').DetectionEvent[] }>('/api/events')
export const getViolations = (cameraId?: number, limit = 100) => {
  const params = new URLSearchParams()
  if (cameraId !== undefined) params.set('camera_id', String(cameraId))
  params.set('limit', String(limit))
  return request<{ violations: import('../types').Violation[]; total_count: number; filtered_count: number }>(
    `/api/violations?${params}`
  )
}
export const getViolationStats = () => request<import('../types').ViolationStats>('/api/violations/stats')
export const exportViolations = (format: 'json' | 'csv' = 'json') => `/api/violations/export?format=${format}`

// ── Audio ──
export const getAudioSettings = () => request<import('../types').AudioSettings>('/api/audio/settings')
export const updateAudioSettings = (settings: Partial<import('../types').AudioSettings>) =>
  request('/api/audio/settings', { method: 'POST', body: JSON.stringify(settings) })
export const testAudio = (soundType?: string) =>
  request('/api/audio/test', { method: 'POST', body: JSON.stringify({ sound_type: soundType }) })

// ── Tamper ──
export const getTamperStatus = () =>
  request<{ tamper_status: Record<number, import('../types').TamperStatus>; config: Record<string, unknown> }>('/api/tamper/status')
export const updateTamperSettings = (settings: Record<string, unknown>) =>
  request('/api/tamper/settings', { method: 'POST', body: JSON.stringify(settings) })

// ── Recordings ──
export const getRecordings = () =>
  request<{ recordings: import('../types').Recording[]; count: number }>('/api/recordings')
export const deleteRecording = (filename: string) =>
  request(`/api/recordings/${filename}`, { method: 'DELETE' })
export const downloadRecording = (filename: string) => `/api/recordings/${filename}`

// ── Reports ──
export const generateReport = (period: 'daily' | 'weekly' | 'monthly' = 'daily') =>
  `/api/reports/generate?period=${period}`
export const getReports = () => request<{ reports: import('../types').Report[] }>('/api/reports/list')

// ── Barrier ──
export const getBarrierSettings = () => request<Record<string, unknown>>('/api/barrier/settings')
export const updateBarrierSettings = (settings: Record<string, unknown>) =>
  request('/api/barrier/settings', { method: 'POST', body: JSON.stringify(settings) })

// ── AI Model Management ──
export interface AIModelStatus {
  available: boolean
  active_model: string
  active_model_info: { description: string; family: string } | null
  description: string
  supported_models: Record<string, { description: string; family: string }>
}
export const getAIModels = () => request<AIModelStatus>('/api/ai/models')
export const switchAIModel = (modelId: string) =>
  request<{ status: string; model: AIModelStatus }>('/api/ai/models/switch', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  })
