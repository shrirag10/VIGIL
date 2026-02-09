import { useVigilStore } from '../stores/vigilStore'
import { AlertTriangle, ShieldAlert } from 'lucide-react'
import clsx from 'clsx'

export default function ViolationTicker() {
  const { violations } = useVigilStore()
  const recent = violations.slice(0, 30)

  return (
    <div className="card h-full flex flex-col min-h-0">
      <div className="flex items-center justify-between mb-2 shrink-0">
        <h3 className="section-title flex items-center gap-1.5">
          <ShieldAlert size={11} className="text-vigil-red" />
          Violations
        </h3>
        <span className="badge-red">{violations.length}</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-1">
        {recent.length > 0 ? (
          recent.map((v, i) => {
            const time = new Date(v.timestamp)
            const isRestricted = v.zone_type === 'restricted'
            const isTamper = v.zone_type === 'tamper'
            return (
              <div
                key={`${v.timestamp}-${i}`}
                className={clsx(
                  'flex items-start gap-2 py-1.5 px-2 rounded-vigil-xs bg-vigil-surface2 border-l-[3px] animate-slide-in',
                  isRestricted && 'border-l-vigil-red',
                  isTamper && 'border-l-vigil-accent',
                  !isRestricted && !isTamper && 'border-l-yellow-500'
                )}
              >
                <AlertTriangle size={11} className={clsx(
                  'mt-0.5 shrink-0',
                  isRestricted ? 'text-vigil-red' : isTamper ? 'text-vigil-accent' : 'text-yellow-400'
                )} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className={clsx(
                      'text-[10px] font-bold uppercase',
                      isRestricted ? 'text-vigil-red' : isTamper ? 'text-vigil-accent' : 'text-yellow-400'
                    )}>
                      {(v.zone_type || 'unknown')}
                    </span>
                    <span className="text-[9px] text-vigil-muted">·</span>
                    <span className="text-[10px] text-vigil-dim truncate">{v.camera_name}</span>
                  </div>
                  <p className="text-[10px] text-vigil-muted truncate">
                    {v.zone_name || v.description || 'Zone violation'}
                  </p>
                  <span className="text-[9px] text-vigil-muted font-mono">{time.toLocaleTimeString()}</span>
                </div>
              </div>
            )
          })
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-vigil-muted">
            <ShieldAlert size={20} className="mb-1 opacity-30" />
            <p className="text-[11px]">No violations recorded</p>
          </div>
        )}
      </div>
    </div>
  )
}
