import { useState } from 'react'
import { useVigilStore } from '../stores/vigilStore'
import CameraFeed from './CameraFeed'
import { Grid2X2, Maximize2, Grid3X3, MonitorPlay } from 'lucide-react'
import clsx from 'clsx'

type Layout = '1x1' | '2x2' | '3x3'

export default function CameraGrid() {
  const { cameras } = useVigilStore()
  const [layout, setLayout] = useState<Layout>('2x2')
  const [focusedCamera, setFocusedCamera] = useState<number | null>(null)

  const gridClass = {
    '1x1': 'grid-cols-1 grid-rows-1',
    '2x2': 'grid-cols-2 grid-rows-2',
    '3x3': 'grid-cols-3 grid-rows-3',
  }

  // Focused view
  if (focusedCamera !== null) {
    const cam = cameras.find((c) => c.id === focusedCamera)
    return (
      <div className="card h-full flex flex-col overflow-hidden">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <MonitorPlay size={14} className="text-vigil-accent" />
            <h3 className="section-title">
              {cam?.position || `Camera ${focusedCamera}`} — Focus
            </h3>
          </div>
          <button onClick={() => setFocusedCamera(null)} className="btn-ghost text-[10px]">
            Back to Grid
          </button>
        </div>
        <div className="flex-1 min-h-0 rounded-vigil-sm overflow-hidden">
          <CameraFeed cameraId={focusedCamera} fullsize />
        </div>
      </div>
    )
  }

  const layoutButtons: { id: Layout; icon: typeof Maximize2; tip: string }[] = [
    { id: '1x1', icon: Maximize2, tip: 'Single' },
    { id: '2x2', icon: Grid2X2,   tip: '2×2' },
    { id: '3x3', icon: Grid3X3,   tip: '3×3' },
  ]

  return (
    <div className="card h-full flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-2 shrink-0">
        <div className="flex items-center gap-2">
          <MonitorPlay size={14} className="text-vigil-accent" />
          <h3 className="section-title">Live Feeds</h3>
          <span className="badge-green">{cameras.filter(c => c.active).length} active</span>
        </div>
        <div className="flex items-center gap-0.5 p-0.5 bg-vigil-bg rounded-vigil-xs border border-vigil-border">
          {layoutButtons.map((btn) => {
            const Icon = btn.icon
            return (
              <button
                key={btn.id}
                onClick={() => setLayout(btn.id)}
                className={clsx(
                  'p-1.5 rounded-[4px] transition-all',
                  layout === btn.id
                    ? 'bg-vigil-surface2 text-vigil-accent shadow-sm'
                    : 'text-vigil-muted hover:text-vigil-text'
                )}
                title={btn.tip}
              >
                <Icon size={13} />
              </button>
            )
          })}
        </div>
      </div>

      {/* Grid */}
      <div className={`flex-1 grid ${gridClass[layout]} gap-2 min-h-0`}>
        {cameras.length > 0
          ? cameras.map((cam) => (
              <CameraFeed
                key={cam.id}
                cameraId={cam.id}
                onFocus={() => setFocusedCamera(cam.id)}
              />
            ))
          : Array.from({ length: layout === '1x1' ? 1 : layout === '2x2' ? 4 : 9 }).map((_, i) => (
              <div key={i} className="rounded-vigil-xs bg-vigil-bg border border-vigil-border flex items-center justify-center">
                <span className="text-[10px] text-vigil-muted font-medium">No Feed</span>
              </div>
            ))}
      </div>
    </div>
  )
}
