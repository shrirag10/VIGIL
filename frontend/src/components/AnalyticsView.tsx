import { useEffect } from 'react'
import { useVigilStore } from '../stores/vigilStore'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, CartesianGrid,
  AreaChart, Area,
} from 'recharts'
import { TrendingUp, Clock, Target, Zap } from 'lucide-react'

const COLORS = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#a855f7']

export default function AnalyticsView() {
  const { violationStats, fetchViolationStats, cameras, violations } = useVigilStore()

  useEffect(() => {
    fetchViolationStats()
    const i = setInterval(fetchViolationStats, 10000)
    return () => clearInterval(i)
  }, [fetchViolationStats])

  const byCameraData = violationStats
    ? Object.entries(violationStats.by_camera).map(([id, count]) => ({ name: `Cam ${id}`, violations: count }))
    : []

  const byZoneData = violationStats
    ? Object.entries(violationStats.by_zone_type).map(([type, count]) => ({ name: type.toUpperCase(), value: count }))
    : []

  const hourlyData = Array.from({ length: 24 }, (_, h) => ({ hour: `${h}:00`, count: 0 }))
  violations.forEach((v) => { hourlyData[new Date(v.timestamp).getHours()].count++ })

  const fpsData = cameras.map((c) => ({ name: c.position || `Cam ${c.id}`, fps: c.fps }))

  const kpis = [
    { label: 'Total Violations', value: violationStats?.total_violations ?? 0, icon: Target, color: 'text-vigil-red', bg: 'bg-vigil-red/10' },
    { label: 'Last Hour', value: violationStats?.violations_last_hour ?? 0, icon: Clock, color: 'text-vigil-accent', bg: 'bg-vigil-accent/10' },
    { label: 'Last 24h', value: violationStats?.violations_last_24h ?? 0, icon: TrendingUp, color: 'text-vigil-blue', bg: 'bg-vigil-blue/10' },
    { label: 'Avg / Hour', value: violationStats?.average_per_hour?.toFixed(1) ?? '0', icon: Zap, color: 'text-vigil-purple', bg: 'bg-vigil-purple/10' },
  ]

  const tooltipStyle = {
    contentStyle: { background: 'var(--v-surface)', border: '1px solid var(--v-border)', borderRadius: 10, fontSize: 11, color: 'var(--v-text)' },
    itemStyle: { color: 'var(--v-text)' },
  }

  return (
    <div className="p-4 h-full overflow-y-auto">
      {/* KPIs */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        {kpis.map((k) => {
          const Icon = k.icon
          return (
            <div key={k.label} className="card flex items-center gap-3">
              <div className={`${k.bg} p-2.5 rounded-vigil-xs`}>
                <Icon size={18} className={k.color} />
              </div>
              <div>
                <p className={`stat-value text-xl ${k.color}`}>{k.value}</p>
                <p className="stat-label">{k.label}</p>
              </div>
            </div>
          )
        })}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-3">
        <ChartCard title="Violations by Camera">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byCameraData}>
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--v-dim)' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--v-dim)' }} axisLine={false} tickLine={false} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="violations" fill="var(--v-red)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="By Zone Type">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={byZoneData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={4} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {byZoneData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip {...tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Hourly Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--v-border)" />
              <XAxis dataKey="hour" tick={{ fontSize: 9, fill: 'var(--v-dim)' }} interval={3} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--v-dim)' }} axisLine={false} tickLine={false} />
              <Tooltip {...tooltipStyle} />
              <Area type="monotone" dataKey="count" stroke="var(--v-accent)" fill="var(--v-accent)" fillOpacity={0.15} strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Camera Performance (FPS)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={fpsData}>
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--v-dim)' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--v-dim)' }} axisLine={false} tickLine={false} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="fps" fill="var(--v-green)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Export */}
      <div className="mt-4 flex gap-2">
        <a href="/api/violations/export?format=json" download className="btn-primary">Export JSON</a>
        <a href="/api/violations/export?format=csv" download className="btn-primary">Export CSV</a>
        <a href="/api/reports/generate?period=daily" target="_blank" rel="noreferrer" className="btn-secondary">Daily Report</a>
        <a href="/api/reports/generate?period=weekly" target="_blank" rel="noreferrer" className="btn-secondary">Weekly Report</a>
      </div>
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card">
      <h3 className="section-title mb-3">{title}</h3>
      {children}
    </div>
  )
}
