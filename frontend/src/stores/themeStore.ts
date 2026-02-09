import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'dark' | 'light' | 'midnight'

interface ThemeState {
  theme: Theme
  setTheme: (theme: Theme) => void
  cycleTheme: () => void
}

const themeOrder: Theme[] = ['dark', 'light', 'midnight']

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      setTheme: (theme) => {
        set({ theme })
        applyTheme(theme)
      },
      cycleTheme: () => {
        const current = get().theme
        const idx = themeOrder.indexOf(current)
        const next = themeOrder[(idx + 1) % themeOrder.length]
        set({ theme: next })
        applyTheme(next)
      },
    }),
    { name: 'vigil-theme' }
  )
)

export function applyTheme(theme: Theme) {
  const root = document.documentElement
  root.classList.remove('dark', 'light', 'midnight')
  root.classList.add(theme)
}

// Apply saved theme on load
const saved = localStorage.getItem('vigil-theme')
if (saved) {
  try {
    const parsed = JSON.parse(saved)
    if (parsed?.state?.theme) {
      applyTheme(parsed.state.theme)
    }
  } catch { /* ignore */ }
}
