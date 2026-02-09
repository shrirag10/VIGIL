import { useVigilStore } from '../stores/vigilStore'
import { PowerOff, Wifi, WifiOff, Activity } from 'lucide-react'
import * as api from '../lib/api'
import ThemeToggle from './ThemeToggle'
import clsx from 'clsx'

export default function Header() {
  const { stats, params } = useVigilStore()
  const isOnline = stats?.system_status === 'operational'

  return (
    <header className="relative flex items-center justify-between px-5 py-2 bg-vigil-surface/90 backdrop-blur-md border-b border-vigil-border z-50">
      {/* Accent strip */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-vigil-accent to-transparent opacity-60" />

      {/* Left: Logo */}
      <div className="flex items-center gap-3">
        <div className="relative w-9 h-9 rounded-vigil-xs bg-gradient-to-br from-vigil-accent via-vigil-red to-vigil-purple grid place-items-center shadow-vigil-glow">
          <span className="text-white font-extrabold text-base leading-none">V</span>
        </div>
        <div className="leading-tight">
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-vigil-text">VIGIL</h1>
            <span className="text-[10px] font-mono font-medium text-vigil-accent bg-vigil-accent/10 px-1.5 py-0.5 rounded">v7.0</span>
          </div>
          <p className="text-[10px] text-vigil-muted leading-none">Vehicle-Installed Guard for Injury Limitation</p>
        </div>
      </div>

      {/* Center: System vitals bar */}
      <div className="flex items-center gap-1.5">
        {params && (
          <div className="flex items-center gap-4 px-4 py-1.5 bg-vigil-surface2 rounded-vigil-sm border border-vigil-border">
            <Metric label="CPU" value={`${params.cpu}%`} warn={params.cpu > 80} />
            <div className="w-px h-4 bg-vigil-border" />
            <Metric label="MEM" value={`${params.memory}%`} warn={params.memory > 85} />
            <div className="w-px h-4 bg-vigil-border" />
            <Metric label="TEMP" value={`${params.temperature}°C`} warn={params.temperature > 75} />
            {stats && (
              <>
                <div className="w-px h-4 bg-vigil-border" />
                <Metric label="FPS" value={(stats.average_fps ?? 0).toFixed(0)} />
              </>
            )}
          </div>
        )}

        <div className={clsx('flex items-center gap-1.5 px-2.5 py-1.5 rounded-vigil-sm border text-[10px] font-bold uppercase tracking-wide',
          isOnline
            ? 'bg-vigil-green/10 border-vigil-green/30 text-vigil-green'
            : 'bg-vigil-red/10 border-vigil-red/30 text-vigil-red'
        )}>
          {isOnline ? <Wifi size={11} /> : <WifiOff size={11} />}
          {isOnline ? 'Online' : 'Offline'}
        </div>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-2">
        {stats && (
          <div className="flex items-center gap-1.5 text-[10px] text-vigil-dim mr-1">
            <Activity size={10} />
            <span className="font-mono">{stats.uptime_formatted}</span>
          </div>
        )}
        <ThemeToggle />
        <button
          onClick={() => {
            if (confirm('Shut down VIGIL system?')) api.shutdownServer()
          }}
          className="btn-icon hover:!border-vigil-red hover:!text-vigil-red hover:!bg-vigil-red/10"
          title="Shutdown"
        >
          <PowerOff size={14} />
        </button>
      </div>
    </header>
  )
}

function Metric({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[9px] font-bold text-vigil-muted uppercase tracking-wider">{label}</span>
      <span className={clsx(
        'text-xs font-mono font-semibold',
        warn ? 'text-vigil-red' : 'text-vigil-text'
      )}>{value}</span>
    </div>
  )
}
