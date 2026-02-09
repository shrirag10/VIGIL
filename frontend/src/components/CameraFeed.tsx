import { useState } from 'react'
import { useVigilStore } from '../stores/vigilStore'
import { getCameraStream } from '../lib/api'
import * as api from '../lib/api'
import { Maximize2, Eye, EyeOff, Video, VideoOff, AlertTriangle } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  cameraId: number
  fullsize?: boolean
  onFocus?: () => void
}

export default function CameraFeed({ cameraId, fullsize, onFocus }: Props) {
  const { cameras, tamperStatus } = useVigilStore()
  const cam = cameras.find((c) => c.id === cameraId)
  const tamper = tamperStatus[cameraId]
  const [imgError, setImgError] = useState(false)

  return (
    <div
      className={clsx(
        'relative bg-vigil-bg rounded-vigil-xs overflow-hidden group flex flex-col border border-vigil-border',
        fullsize && 'h-full',
        tamper?.is_tampered && 'tamper-warning',
        cam?.recording && 'ring-2 ring-vigil-red/60'
      )}
    >
      {/* Top-left: camera label */}
      <div className="absolute top-0 left-0 z-10 flex items-center gap-1.5 px-2 py-1 bg-black/60 backdrop-blur-sm rounded-br-lg">
        <span className={clsx('status-dot', cam?.active ? 'status-dot-active' : 'status-dot-inactive')} />
        <span className="text-[10px] font-semibold text-white/90">{cam?.position || `Cam ${cameraId}`}</span>
      </div>

      {/* Top-right: FPS + detection count */}
      <div className="absolute top-0 right-0 z-10 flex items-center gap-2 px-2 py-1 bg-black/60 backdrop-blur-sm rounded-bl-lg">
        <span className="text-[9px] font-mono text-vigil-green">{cam?.fps ?? 0} fps</span>
        <span className="text-[9px] font-mono text-vigil-accent">{cam?.detections ?? 0} det</span>
      </div>

      {/* Video feed */}
      <div className="relative flex-1 min-h-0">
        {!imgError ? (
          <img
            src={getCameraStream(cameraId)}
            alt={`Camera ${cameraId}`}
            className="w-full h-full object-contain"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-1">
            <VideoOff size={18} className="text-vigil-muted" />
            <span className="text-[10px] text-vigil-muted">Feed unavailable</span>
          </div>
        )}

        {/* Tamper overlay */}
        {tamper?.is_tampered && (
          <div className="absolute inset-0 bg-red-900/70 backdrop-blur-sm flex items-center justify-center">
            <div className="text-center animate-blink">
              <AlertTriangle size={28} className="mx-auto text-white" />
              <p className="text-sm font-bold text-white mt-1.5">TAMPER: {tamper.tamper_type}</p>
              <p className="text-[10px] text-red-200">Check camera immediately</p>
            </div>
          </div>
        )}

        {/* Recording badge */}
        {cam?.recording && (
          <div className="absolute top-8 right-2 flex items-center gap-1 px-2 py-0.5 rounded-full bg-vigil-red text-white text-[9px] font-bold recording-active">
            <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
            REC
          </div>
        )}

        {/* Hover controls */}
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-2.5 pt-8 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <div className="flex items-end justify-between">
            <div>
              <p className="text-[11px] font-semibold text-white">{cam?.position || `Camera ${cameraId}`}</p>
              <p className="text-[9px] text-gray-400 font-mono">{cam?.resolution || '—'}</p>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => cam && api.toggleDetection(cameraId, !cam.detection_enabled)}
                className={clsx(
                  'p-1.5 rounded-vigil-xs transition-all',
                  cam?.detection_enabled ? 'bg-vigil-green/30 text-vigil-green' : 'bg-white/10 text-white/50'
                )}
                title={cam?.detection_enabled ? 'Disable detection' : 'Enable detection'}
              >
                {cam?.detection_enabled ? <Eye size={13} /> : <EyeOff size={13} />}
              </button>
              <button
                onClick={() => cam && api.toggleRecording(cameraId, !cam.recording)}
                className={clsx(
                  'p-1.5 rounded-vigil-xs transition-all',
                  cam?.recording ? 'bg-vigil-red/30 text-vigil-red' : 'bg-white/10 text-white/50'
                )}
                title={cam?.recording ? 'Stop recording' : 'Start recording'}
              >
                {cam?.recording ? <VideoOff size={13} /> : <Video size={13} />}
              </button>
              {onFocus && (
                <button
                  onClick={onFocus}
                  className="p-1.5 rounded-vigil-xs bg-white/10 text-white/50 hover:text-white transition-all"
                  title="Focus view"
                >
                  <Maximize2 size={13} />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
