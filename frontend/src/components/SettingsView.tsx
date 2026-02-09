import { useEffect, useState } from 'react'
import { useVigilStore } from '../stores/vigilStore'
import { useThemeStore, type Theme } from '../stores/themeStore'
import * as api from '../lib/api'
import { playAlarm } from '../lib/audioEngine'
import { Volume2, Shield, AlertTriangle, Cpu, Bell, Palette, Sun, Moon, Monitor, BrainCircuit, Loader2 } from 'lucide-react'
import clsx from 'clsx'

const themes: { id: Theme; label: string; icon: typeof Sun; desc: string; colors: string[] }[] = [
  { id: 'dark',     label: 'Dark',     icon: Moon,    desc: 'Deep charcoal with amber accents',   colors: ['#0b0b11', '#131320', '#f59e0b'] },
  { id: 'light',    label: 'Light',    icon: Sun,     desc: 'Clean white with warm highlights',   colors: ['#f0f2f5', '#ffffff', '#d97706'] },
  { id: 'midnight', label: 'Midnight', icon: Monitor, desc: 'Navy slate with cyan highlights',    colors: ['#020617', '#0f172a', '#38bdf8'] },
]

export default function SettingsView() {
  const { audioSettings, fetchAudioSettings } = useVigilStore()
  const { theme, setTheme } = useThemeStore()
  const [barrierSettings, setBarrierSettings] = useState<Record<string, unknown>>({})
  const [tamperConfig, setTamperConfig] = useState<Record<string, unknown>>({})
  const [aiModels, setAiModels] = useState<api.AIModelStatus | null>(null)
  const [switchingModel, setSwitchingModel] = useState(false)

  useEffect(() => {
    fetchAudioSettings()
    api.getBarrierSettings().then(setBarrierSettings).catch(() => {})
    api.getTamperStatus().then((d) => setTamperConfig(d.config)).catch(() => {})
    api.getAIModels().then(setAiModels).catch(() => {})
  }, [fetchAudioSettings])

  const handleAudioToggle = async () => {
    if (!audioSettings) return
    await api.updateAudioSettings({ enabled: !audioSettings.enabled })
    fetchAudioSettings()
  }

  const handleVolumeChange = async (volume: number) => {
    await api.updateAudioSettings({ volume })
    fetchAudioSettings()
  }

  const handleBarrierUpdate = async (key: string, value: unknown) => {
    setBarrierSettings(p => ({ ...p, [key]: value }))
    await api.updateBarrierSettings({ [key]: value })
  }

  const handleTamperUpdate = async (key: string, value: unknown) => {
    setTamperConfig(p => ({ ...p, [key]: value }))
    await api.updateTamperSettings({ [key]: value })
  }

  const handleModelSwitch = async (modelId: string) => {
    if (switchingModel || modelId === aiModels?.active_model) return
    setSwitchingModel(true)
    try {
      const res = await api.switchAIModel(modelId)
      setAiModels(res.model)
    } catch (err) {
      console.error('Model switch failed:', err)
    } finally {
      setSwitchingModel(false)
    }
  }

  return (
    <div className="p-4 h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto space-y-4">

        {/* ── Theme Selector ── */}
        <div className="card">
          <h3 className="text-sm font-semibold flex items-center gap-2 mb-4">
            <Palette size={16} className="text-vigil-accent" />
            Appearance
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {themes.map((t) => {
              const Icon = t.icon
              const isActive = theme === t.id
              return (
                <button
                  key={t.id}
                  onClick={() => setTheme(t.id)}
                  className={clsx(
                    'relative p-4 rounded-vigil-sm border-2 text-left transition-all duration-200',
                    isActive
                      ? 'border-vigil-accent bg-vigil-accent/5 shadow-vigil-glow'
                      : 'border-vigil-border hover:border-vigil-border-light bg-vigil-surface2'
                  )}
                >
                  {isActive && (
                    <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-vigil-accent animate-pulse" />
                  )}
                  <div className="flex items-center gap-2 mb-2">
                    <Icon size={16} className={isActive ? 'text-vigil-accent' : 'text-vigil-dim'} />
                    <span className="text-sm font-semibold">{t.label}</span>
                  </div>
                  <p className="text-[10px] text-vigil-dim mb-3">{t.desc}</p>
                  <div className="flex gap-1">
                    {t.colors.map((c, i) => (
                      <span key={i} className="w-5 h-5 rounded-full border border-vigil-border shadow-vigil-inset" style={{ background: c }} />
                    ))}
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* ── AI Detection Model ── */}
        <div className="card">
          <h3 className="text-sm font-semibold flex items-center gap-2 mb-4">
            <BrainCircuit size={16} className="text-vigil-accent" />
            AI Detection Model
            {switchingModel && <Loader2 size={14} className="animate-spin text-vigil-accent ml-auto" />}
          </h3>
          {aiModels ? (
            <>
              <p className="text-xs text-vigil-dim mb-3">
                Active: <span className="text-vigil-text font-medium">{aiModels.active_model}</span>
                {aiModels.active_model_info && (
                  <span className="ml-1.5 text-vigil-dim">— {aiModels.active_model_info.description}</span>
                )}
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {Object.entries(aiModels.supported_models).map(([id, info]) => {
                  const isActive = id === aiModels.active_model
                  return (
                    <button
                      key={id}
                      onClick={() => handleModelSwitch(id)}
                      disabled={switchingModel}
                      className={`relative p-3 rounded-lg border text-left transition-all ${
                        isActive
                          ? 'border-vigil-accent ring-1 ring-vigil-accent'
                          : 'border-vigil-border hover:border-vigil-accent bg-vigil-card'
                      } ${switchingModel ? 'opacity-60 cursor-wait' : ''}`}
                      style={isActive ? { background: 'color-mix(in srgb, var(--v-accent) 10%, transparent)' } : undefined}
                    >
                      {isActive && (
                        <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-vigil-green" />
                      )}
                      <span className="text-xs font-bold block mb-0.5">{id}</span>
                      <span className="text-[10px] text-vigil-dim block">{info.description}</span>
                      <span className="text-[9px] text-vigil-dim mt-1 block uppercase tracking-wide">{info.family}</span>
                    </button>
                  )
                })}
              </div>
            </>
          ) : (
            <p className="text-xs text-vigil-dim">Loading models…</p>
          )}
        </div>

        {/* ── Audio ── */}
        <div className="card">
          <h3 className="text-sm font-semibold flex items-center gap-2 mb-4">
            <Volume2 size={16} className="text-vigil-accent" />
            Audio Alarm Settings
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="stat-label block mb-1.5">Enabled</label>
              <Toggle checked={audioSettings?.enabled ?? false} onToggle={handleAudioToggle} />
            </div>
            <div>
              <label className="stat-label block mb-1.5">Volume: {((audioSettings?.volume ?? 0.7) * 100).toFixed(0)}%</label>
              <input type="range" min="0" max="1" step="0.05" value={audioSettings?.volume ?? 0.7}
                onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
                className="w-full accent-vigil-accent h-1.5"
              />
            </div>
            <div>
              <label className="stat-label block mb-1.5">Sound Type</label>
              <select value={audioSettings?.sound_type ?? 'alert'}
                onChange={(e) => api.updateAudioSettings({ sound_type: e.target.value as 'alert' | 'siren' | 'beep' }).then(fetchAudioSettings)}
                className="select"
              >
                <option value="alert">Alert</option>
                <option value="siren">Siren</option>
                <option value="beep">Beep</option>
              </select>
            </div>
            <div className="flex items-end">
              <button onClick={() => {
                playAlarm(audioSettings?.sound_type ?? 'alert', audioSettings?.volume ?? 0.7)
                api.testAudio(audioSettings?.sound_type)
              }} className="btn-primary">
                <Bell size={14} /> Test Alarm
              </button>
            </div>
          </div>
        </div>

        {/* ── Barrier ── */}
        <div className="card">
          <h3 className="text-sm font-semibold flex items-center gap-2 mb-4">
            <Shield size={16} className="text-vigil-green" />
            Barrier Detection
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {[
              { key: 'enabled', label: 'Enabled', type: 'toggle' },
              { key: 'min_area', label: 'Min Area', type: 'number' },
              { key: 'max_area', label: 'Max Area', type: 'number' },
              { key: 'min_width', label: 'Min Width', type: 'number' },
              { key: 'min_height', label: 'Min Height', type: 'number' },
              { key: 'min_saturation', label: 'Min Saturation', type: 'number' },
            ].map((s) => (
              <div key={s.key}>
                <label className="stat-label block mb-1.5">{s.label}</label>
                {s.type === 'toggle' ? (
                  <Toggle checked={!!barrierSettings[s.key]} onToggle={() => handleBarrierUpdate(s.key, !barrierSettings[s.key])} />
                ) : (
                  <input type="number" value={String(barrierSettings[s.key] ?? '')}
                    onChange={(e) => handleBarrierUpdate(s.key, parseInt(e.target.value))} className="input" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ── Tamper ── */}
        <div className="card">
          <h3 className="text-sm font-semibold flex items-center gap-2 mb-4">
            <AlertTriangle size={16} className="text-vigil-red" />
            Tamper Detection
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {[
              { key: 'enabled', label: 'Enabled', type: 'toggle' },
              { key: 'darkness_threshold', label: 'Darkness Threshold', type: 'number' },
              { key: 'warning_delay', label: 'Warning Delay (sec)', type: 'number' },
            ].map((s) => (
              <div key={s.key}>
                <label className="stat-label block mb-1.5">{s.label}</label>
                {s.type === 'toggle' ? (
                  <Toggle checked={!!tamperConfig[s.key]} onToggle={() => handleTamperUpdate(s.key, !tamperConfig[s.key])} />
                ) : (
                  <input type="number" value={String(tamperConfig[s.key] ?? '')}
                    onChange={(e) => handleTamperUpdate(s.key, parseInt(e.target.value))} className="input" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ── System Info ── */}
        <div className="card">
          <h3 className="text-sm font-semibold flex items-center gap-2 mb-4">
            <Cpu size={16} className="text-vigil-blue" />
            System Information
          </h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[
              ['Version', 'VIGIL V7.0'],
              ['Architecture', 'React + FastAPI + gRPC + Kafka'],
              ['Detection', `${aiModels?.active_model ?? 'YOLOv8n'} (person + barrier)`],
              ['Frontend', 'React 18 + Tailwind + Recharts'],
            ].map(([k, v]) => (
              <div key={k} className="p-2.5 bg-vigil-surface2 rounded-vigil-xs">
                <span className="text-vigil-dim">{k}: </span>
                <span className="font-medium font-mono">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function Toggle({ checked, onToggle }: { checked: boolean; onToggle: () => void }) {
  return (
    <button onClick={onToggle}
      className={clsx('toggle', checked ? 'bg-vigil-green' : 'bg-vigil-border')}
    >
      <span className={clsx('toggle-thumb', checked ? 'left-[22px]' : 'left-0.5')} />
    </button>
  )
}
