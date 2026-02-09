import { useVigilStore } from '../stores/vigilStore'
import { Eye, Shield, Cpu } from 'lucide-react'
import clsx from 'clsx'

export default function StatsPanel() {
  const { stats, cameras } = useVigilStore()

  return (
    <div className="card h-full flex items-stretch gap-3 overflow-hidden">
      {/* Camera status list */}
      <div className="flex-1 min-w-0">
        <h3 className="section-title mb-2 flex items-center gap-1.5">
          <Eye size={11} className="text-vigil-blue" />
          Camera Status
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {cameras.map((cam) => (
            <div
              key={cam.id}
              className={clsx(
                'flex items-center gap-2 px-3 py-1.5 rounded-vigil-xs border transition-colors',
                cam.active
                  ? 'border-vigil-green bg-vigil-surface2'
                  : 'border-vigil-red bg-vigil-surface2'
              )}
            >
              <span className={clsx('status-dot', cam.active ? 'status-dot-active' : 'status-dot-inactive')} />
              <div className="min-w-0">
                <span className="text-[11px] font-semibold block truncate">{cam.position || cam.name}</span>
                <span className="text-[9px] text-vigil-dim font-mono">
                  {cam.fps} fps · {cam.detections} det
                  {cam.recording && <span className="text-vigil-red ml-1">REC</span>}
                </span>
              </div>
            </div>
          ))}
          {cameras.length === 0 && (
            <span className="text-[11px] text-vigil-muted">No cameras</span>
          )}
        </div>
      </div>

      {/* Divider */}
      <div className="w-px bg-vigil-border shrink-0" />

      {/* AI Engine + Architecture info */}
      <div className="shrink-0 w-52 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <Cpu size={12} className="text-vigil-accent" />
          <span className="section-title">AI Engine</span>
        </div>
        <div className="flex items-center gap-2 px-2 py-1.5 rounded-vigil-xs bg-vigil-surface2">
          <span className={clsx('status-dot', stats?.yolo_available ? 'status-dot-active' : 'status-dot-inactive')} />
          <span className="text-[11px] font-semibold">
            {stats?.active_model || 'N/A'} {stats?.yolo_available ? 'Active' : 'Off'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Shield size={12} className="text-vigil-green" />
          <span className="section-title">Architecture</span>
        </div>
        <div className="flex flex-wrap gap-1">
          {['React', 'FastAPI', 'gRPC', 'Kafka'].map(t => (
            <span key={t} className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-vigil-surface2 text-vigil-dim border border-vigil-border">{t}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
