import { useThemeStore, type Theme } from '../stores/themeStore'
import { Sun, Moon, Monitor } from 'lucide-react'

const themeInfo: Record<Theme, { icon: typeof Sun; label: string }> = {
  dark:     { icon: Moon,    label: 'Dark' },
  light:    { icon: Sun,     label: 'Light' },
  midnight: { icon: Monitor, label: 'Midnight' },
}

export default function ThemeToggle() {
  const { theme, cycleTheme } = useThemeStore()
  const { icon: Icon, label } = themeInfo[theme]

  return (
    <button
      onClick={cycleTheme}
      className="btn-icon group relative"
      title={`Theme: ${label}`}
    >
      <Icon size={15} className="transition-transform group-hover:rotate-45" />
      <span className="absolute -bottom-7 left-1/2 -translate-x-1/2 text-[9px] font-semibold
                       text-vigil-dim opacity-0 group-hover:opacity-100 transition-opacity
                       whitespace-nowrap pointer-events-none">
        {label}
      </span>
    </button>
  )
}
