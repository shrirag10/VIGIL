import { useVigilStore } from '../stores/vigilStore'
import type { TabId } from '../types'
import { LayoutDashboard, BarChart3, MapPin, Video, Settings } from 'lucide-react'
import clsx from 'clsx'

const tabs: { id: TabId; label: string; icon: typeof LayoutDashboard; accent: string }[] = [
  { id: 'dashboard',  label: 'Dashboard',  icon: LayoutDashboard, accent: 'text-vigil-accent' },
  { id: 'analytics',  label: 'Analytics',   icon: BarChart3,       accent: 'text-vigil-blue' },
  { id: 'zones',      label: 'Zones',       icon: MapPin,          accent: 'text-vigil-green' },
  { id: 'recordings', label: 'Recordings',  icon: Video,           accent: 'text-vigil-purple' },
  { id: 'settings',   label: 'Settings',    icon: Settings,        accent: 'text-vigil-dim' },
]

export default function TabNav() {
  const { activeTab, setActiveTab } = useVigilStore()

  return (
    <nav className="flex items-center gap-0.5 px-5 py-1.5 bg-vigil-bg border-b border-vigil-border">
      {tabs.map((tab) => {
        const Icon = tab.icon
        const isActive = activeTab === tab.id
        return (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={clsx(
              'relative flex items-center gap-2 px-4 py-2 rounded-vigil-xs text-xs font-semibold transition-all duration-200',
              isActive
                ? 'bg-vigil-surface text-vigil-text shadow-vigil border border-vigil-border'
                : 'text-vigil-dim hover:text-vigil-text-secondary hover:bg-vigil-surface/50 border border-transparent'
            )}
          >
            <Icon size={14} className={isActive ? tab.accent : ''} />
            {tab.label}
            {isActive && (
              <span className={clsx(
                'absolute -bottom-[7px] left-1/2 -translate-x-1/2 w-6 h-[3px] rounded-full',
                tab.id === 'dashboard' ? 'bg-vigil-accent' :
                tab.id === 'analytics' ? 'bg-vigil-blue' :
                tab.id === 'zones' ? 'bg-vigil-green' :
                tab.id === 'recordings' ? 'bg-vigil-purple' : 'bg-vigil-dim'
              )} />
            )}
          </button>
        )
      })}
    </nav>
  )
}
