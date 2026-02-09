import { useState, useEffect, useRef, useCallback } from 'react'
import { useVigilStore } from '../stores/vigilStore'
import * as api from '../lib/api'
import { getCameraStream } from '../lib/api'
import type { Zone } from '../types'
import { Plus, Trash2, MapPin } from 'lucide-react'
import clsx from 'clsx'

export default function ZoneManagement() {
  const { cameras } = useVigilStore()
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null)
  const [zones, setZones] = useState<Zone[]>([])
  const [drawing, setDrawing] = useState(false)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [drawEnd, setDrawEnd] = useState<{ x: number; y: number } | null>(null)
  const [newZoneType, setNewZoneType] = useState<'restricted' | 'warning' | 'safe'>('restricted')
  const [newZoneName, setNewZoneName] = useState('')
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    if (cameras.length > 0 && selectedCamera === null) setSelectedCamera(cameras[0].id)
  }, [cameras, selectedCamera])

  useEffect(() => {
    if (selectedCamera === null) return
    api.getCameraZones(selectedCamera).then((data) => setZones(data.zones)).catch(() => {})
  }, [selectedCamera])

  const getRelativePos = useCallback((e: React.MouseEvent) => {
    if (!imgRef.current) return { x: 0, y: 0 }
    const rect = imgRef.current.getBoundingClientRect()
    return {
      x: Math.round(((e.clientX - rect.left) / rect.width) * 480),
      y: Math.round(((e.clientY - rect.top) / rect.height) * 360),
    }
  }, [])

  const handleMouseDown = (e: React.MouseEvent) => { if (!drawing) return; setDrawStart(getRelativePos(e)); setDrawEnd(null) }
  const handleMouseMove = (e: React.MouseEvent) => { if (!drawing || !drawStart) return; setDrawEnd(getRelativePos(e)) }

  const handleMouseUp = async () => {
    if (!drawing || !drawStart || !drawEnd || selectedCamera === null) return
    const rect: [number, number, number, number] = [
      Math.min(drawStart.x, drawEnd.x), Math.min(drawStart.y, drawEnd.y),
      Math.max(drawStart.x, drawEnd.x), Math.max(drawStart.y, drawEnd.y),
    ]
    try {
      await api.addCameraZone(selectedCamera, { name: newZoneName || `${newZoneType} zone`, type: newZoneType, rect })
      const data = await api.getCameraZones(selectedCamera)
      setZones(data.zones)
    } catch (err) { console.error('Failed to create zone', err) }
    setDrawing(false); setDrawStart(null); setDrawEnd(null); setNewZoneName('')
  }

  const handleDeleteZone = async (zoneId: string) => {
    if (selectedCamera === null) return
    try {
      await api.deleteCameraZone(selectedCamera, zoneId)
      setZones((z) => z.filter((zone) => zone.id !== zoneId))
    } catch (err) { console.error('Failed to delete zone', err) }
  }

  const zoneTypeColors: Record<string, string> = {
    restricted: 'border-vigil-red bg-vigil-red/5',
    warning: 'border-yellow-500 bg-yellow-500/5',
    safe: 'border-vigil-green bg-vigil-green/5',
  }

  return (
    <div className="p-4 h-full overflow-y-auto">
      <div className="grid grid-cols-[1fr_300px] gap-3 h-full">
        {/* Left - Camera feed with zone overlay */}
        <div className="card flex flex-col">
          <div className="flex items-center gap-2 mb-3">
            <MapPin size={14} className="text-vigil-accent" />
            <h3 className="text-sm font-semibold">Zone Editor</h3>
            <div className="flex gap-1 ml-auto">
              {cameras.map((cam) => (
                <button
                  key={cam.id}
                  onClick={() => setSelectedCamera(cam.id)}
                  className={clsx(
                    'px-3 py-1 rounded-vigil-xs text-xs font-medium transition-all',
                    selectedCamera === cam.id
                      ? 'bg-vigil-accent text-black'
                      : 'bg-vigil-surface2 text-vigil-dim hover:text-vigil-text'
                  )}
                >
                  {cam.position || cam.name}
                </button>
              ))}
            </div>
          </div>

          <div
            className="relative flex-1 bg-vigil-bg rounded-vigil-sm overflow-hidden cursor-crosshair border border-vigil-border"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
          >
            {selectedCamera !== null && (
              <img ref={imgRef} src={getCameraStream(selectedCamera)} alt="Zone editor feed" className="w-full h-full object-contain" />
            )}
            {drawing && drawStart && drawEnd && (
              <div
                className={clsx(
                  'absolute border-2 pointer-events-none',
                  newZoneType === 'restricted' ? 'border-vigil-red bg-vigil-red/20' :
                  newZoneType === 'warning' ? 'border-yellow-500 bg-yellow-500/20' :
                  'border-vigil-green bg-vigil-green/20'
                )}
                style={{
                  left: `${(Math.min(drawStart.x, drawEnd.x) / 480) * 100}%`,
                  top: `${(Math.min(drawStart.y, drawEnd.y) / 360) * 100}%`,
                  width: `${(Math.abs(drawEnd.x - drawStart.x) / 480) * 100}%`,
                  height: `${(Math.abs(drawEnd.y - drawStart.y) / 360) * 100}%`,
                }}
              />
            )}
            {!drawing && (
              <div className="absolute bottom-3 left-3 text-[10px] text-white/70 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-vigil-xs">
                Click "Draw Zone" then drag on feed
              </div>
            )}
          </div>
        </div>

        {/* Right - Zone controls */}
        <div className="flex flex-col gap-3">
          <div className="card">
            <h4 className="section-title mb-2">Create Zone</h4>
            <input type="text" placeholder="Zone name..." value={newZoneName}
              onChange={(e) => setNewZoneName(e.target.value)} className="input mb-2" />
            <div className="flex gap-1 mb-2">
              {(['restricted', 'warning', 'safe'] as const).map((type) => (
                <button key={type} onClick={() => setNewZoneType(type)}
                  className={clsx(
                    'flex-1 py-1.5 rounded-vigil-xs text-[10px] font-bold uppercase transition-all',
                    newZoneType === type
                      ? type === 'restricted' ? 'bg-vigil-red text-white' : type === 'warning' ? 'bg-yellow-500 text-black' : 'bg-vigil-green text-white'
                      : 'bg-vigil-surface2 text-vigil-dim'
                  )}
                >{type}</button>
              ))}
            </div>
            <button onClick={() => setDrawing(true)} disabled={drawing} className="btn-primary w-full">
              <Plus size={14} />
              {drawing ? 'Drawing... drag on feed' : 'Draw Zone'}
            </button>
          </div>

          <div className="card flex-1 overflow-y-auto">
            <h4 className="section-title mb-2">Active Zones ({zones.length})</h4>
            <div className="space-y-1.5">
              {zones.map((zone) => (
                <div key={zone.id} className={clsx('flex items-center justify-between p-2 rounded-vigil-xs border', zoneTypeColors[zone.type] || 'border-vigil-border')}>
                  <div>
                    <p className="text-xs font-medium">{zone.name || 'Unnamed'}</p>
                    <p className="text-[10px] text-vigil-dim uppercase">{zone.type}</p>
                  </div>
                  <button onClick={() => handleDeleteZone(zone.id)} className="p-1 rounded hover:bg-vigil-red/20 text-vigil-dim hover:text-vigil-red transition-all">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
              {zones.length === 0 && <p className="text-xs text-vigil-muted text-center py-4">No zones configured</p>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
