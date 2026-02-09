import { useVigilStore } from '../stores/vigilStore'
import { Activity } from 'lucide-react'

export default function EventLog() {
  const { events } = useVigilStore()
  const recent = events.slice(-20).reverse()

  return (
    <div className="card h-full flex flex-col min-h-0">
      <div className="flex items-center justify-between mb-2 shrink-0">
        <h3 className="section-title flex items-center gap-1.5">
          <Activity size={11} className="text-vigil-blue" />
          Event Log
        </h3>
        <span className="badge-blue">{events.length}</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-1">
        {recent.length > 0 ? (
          recent.map((evt, i) => {
            const time = new Date(evt.timestamp)
            return (
              <div key={`${evt.timestamp}-${i}`} className="py-1.5 px-2 rounded-vigil-xs bg-vigil-surface2">
                <p className="text-[11px] font-medium truncate text-vigil-text">{evt.description}</p>
                <p className="text-[9px] text-vigil-muted font-mono">
                  {evt.event_type} · Cam {evt.camera_id} · {time.toLocaleTimeString()}
                </p>
              </div>
            )
          })
        ) : (
          <div className="flex flex-col items-center justify-center py-6 text-vigil-muted">
            <Activity size={18} className="mb-1 opacity-30" />
            <p className="text-[11px]">No events</p>
          </div>
        )}
      </div>
    </div>
  )
}
