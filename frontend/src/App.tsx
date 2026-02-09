import { useEffect, useRef } from 'react'
import { useVigilStore } from './stores/vigilStore'
import { playAlarm } from './lib/audioEngine'
import Header from './components/Header'
import TabNav from './components/TabNav'
import DashboardView from './components/DashboardView'
import AnalyticsView from './components/AnalyticsView'
import ZoneManagement from './components/ZoneManagement'
import RecordingsView from './components/RecordingsView'
import SettingsView from './components/SettingsView'

/** Play alarm when new violations arrive */
function useViolationAlarm() {
  const violations = useVigilStore((s) => s.violations)
  const audioSettings = useVigilStore((s) => s.audioSettings)
  const prevCount = useRef(violations.length)

  useEffect(() => {
    if (violations.length > prevCount.current && audioSettings?.enabled) {
      playAlarm(audioSettings.sound_type, audioSettings.volume)
    }
    prevCount.current = violations.length
  }, [violations.length, audioSettings])
}

export default function App() {
  const { activeTab, startPolling, stopPolling } = useVigilStore()

  useEffect(() => {
    startPolling()
    return () => stopPolling()
  }, [startPolling, stopPolling])

  useViolationAlarm()

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header />
      <TabNav />
      <main className="flex-1 overflow-hidden">
        {activeTab === 'dashboard' && <DashboardView />}
        {activeTab === 'analytics' && <AnalyticsView />}
        {activeTab === 'zones' && <ZoneManagement />}
        {activeTab === 'recordings' && <RecordingsView />}
        {activeTab === 'settings' && <SettingsView />}
      </main>
    </div>
  )
}
