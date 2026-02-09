// ── Camera ──
export interface Camera {
  id: number
  name: string
  position: string
  device_index: number
  active: boolean
  recording: boolean
  detections: number
  violations: number
  fps: number
  total_detections: number
  status: string
  resolution: string
  detection_enabled: boolean
}

// ── Zone ──
export interface Zone {
  id: string
  camera_id: number
  name: string
  type: 'restricted' | 'warning' | 'safe'
  rect?: [number, number, number, number]
  points?: number[][]
  active: boolean
  created_at?: string
}

// ── Violation ──
export interface Violation {
  timestamp: string
  camera_id: number
  camera_name: string
  zone_id: string
  zone_name: string
  zone_type: 'restricted' | 'warning' | 'tamper'
  person_box?: number[]
  tamper_type?: string
  description?: string
}

// ── Events ──
export interface DetectionEvent {
  timestamp: string
  camera_id: number
  event_type: string
  description: string
  confidence: number
}

// ── System Stats ──
export interface SystemStats {
  uptime_seconds: number
  uptime_formatted: string
  total_cameras: number
  active_cameras: number
  total_detections: number
  total_violations: number
  peak_detections: number
  average_fps: number
  detection_rate_per_minute: number
  yolo_available: boolean
  active_model: string
  system_status: string
}

// ── Violation Stats ──
export interface ViolationStats {
  total_violations: number
  violations_last_hour: number
  violations_last_24h: number
  by_camera: Record<number, number>
  by_zone_type: Record<string, number>
  average_per_hour: number
}

// ── System Params ──
export interface SystemParams {
  cpu: number
  memory: number
  temperature: number
}

// ── Audio Settings ──
export interface AudioSettings {
  enabled: boolean
  volume: number
  sound_type: 'alert' | 'siren' | 'beep'
  repeat_interval: number
  cooldown: number
}

// ── Tamper Status ──
export interface TamperStatus {
  is_tampered: boolean
  tamper_type: string | null
  warning_sent: boolean
  camera_name: string
}

// ── Recording ──
export interface Recording {
  filename: string
  size_mb: number
  created: string
  path: string
}

// ── Report ──
export interface Report {
  filename: string
  size: number
  created: string
  download_url: string
}

// ── Tab ──
export type TabId = 'dashboard' | 'analytics' | 'zones' | 'recordings' | 'settings'
