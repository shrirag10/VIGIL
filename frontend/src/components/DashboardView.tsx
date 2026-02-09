import { useVigilStore } from '../stores/vigilStore'
import CameraGrid from './CameraGrid'
import StatsPanel from './StatsPanel'
import ViolationTicker from './ViolationTicker'
import EventLog from './EventLog'

export default function DashboardView() {
  const { stats } = useVigilStore()

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Top: KPI ribbon */}
      <div className="shrink-0 grid grid-cols-6 gap-2 px-3 pt-3">
        <KpiCard label="Active Cameras" value={`${stats?.active_cameras ?? 0}/${stats?.total_cameras ?? 0}`} color="text-vigil-blue" />
        <KpiCard label="Live Detections" value={String(stats?.total_detections ?? 0)} color="text-vigil-green" />
        <KpiCard label="Violations" value={String(stats?.total_violations ?? 0)} color="text-vigil-red" />
        <KpiCard label="Avg FPS" value={(stats?.average_fps ?? 0).toFixed(1)} color="text-vigil-accent" />
        <KpiCard label="Peak Det." value={String(stats?.peak_detections ?? 0)} color="text-vigil-purple" />
        <KpiCard label="Det/Min" value={(stats?.detection_rate_per_minute ?? 0).toFixed(1)} color="text-vigil-cyan" />
      </div>

      {/* Main: Camera + side panels */}
      <div className="flex-1 grid grid-cols-[1fr_300px] gap-2 p-3 min-h-0 overflow-hidden">
        {/* Left: cameras above, stats below */}
        <div className="flex flex-col gap-2 min-h-0 overflow-hidden">
          <div className="flex-[3] min-h-0">
            <CameraGrid />
          </div>
          <div className="flex-[1] min-h-0">
            <StatsPanel />
          </div>
        </div>

        {/* Right: Violations + Events stacked */}
        <div className="flex flex-col gap-2 min-h-0 overflow-hidden">
          <div className="flex-[2] min-h-0">
            <ViolationTicker />
          </div>
          <div className="flex-1 min-h-0">
            <EventLog />
          </div>
        </div>
      </div>
    </div>
  )
}

function KpiCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="card-sm flex flex-col items-center gap-0.5 py-2 text-center">
      <span className={`text-lg font-bold font-mono tabular-nums tracking-tight ${color}`}>{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}
