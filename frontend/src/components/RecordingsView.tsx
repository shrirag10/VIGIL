import { useEffect } from 'react'
import { useVigilStore } from '../stores/vigilStore'
import * as api from '../lib/api'
import { Video, Trash2, Download, FileText } from 'lucide-react'

export default function RecordingsView() {
  const { recordings, fetchRecordings, reports, fetchReports } = useVigilStore()

  useEffect(() => { fetchRecordings(); fetchReports() }, [fetchRecordings, fetchReports])

  const handleDelete = async (filename: string) => {
    if (!confirm(`Delete recording "${filename}"?`)) return
    try { await api.deleteRecording(filename); fetchRecordings() } catch (err) { console.error('Failed to delete', err) }
  }

  return (
    <div className="p-4 h-full overflow-y-auto">
      <div className="grid grid-cols-2 gap-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Video size={16} className="text-vigil-accent" />
              Video Recordings
            </h3>
            <span className="badge-blue">{recordings.length} files</span>
          </div>
          <div className="space-y-1.5 max-h-[calc(100vh-240px)] overflow-y-auto">
            {recordings.map((rec) => (
              <div key={rec.filename} className="flex items-center justify-between p-2.5 rounded-vigil-xs bg-vigil-surface2">
                <div>
                  <p className="text-xs font-medium">{rec.filename}</p>
                  <p className="text-[10px] text-vigil-dim font-mono">{rec.size_mb} MB · {new Date(rec.created).toLocaleString()}</p>
                </div>
                <div className="flex items-center gap-1">
                  <a href={api.downloadRecording(rec.filename)} download className="btn-icon w-7 h-7" title="Download">
                    <Download size={13} />
                  </a>
                  <button onClick={() => handleDelete(rec.filename)} className="btn-icon w-7 h-7 hover:!border-vigil-red hover:!text-vigil-red" title="Delete">
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
            {recordings.length === 0 && <p className="text-xs text-vigil-muted text-center py-8">No recordings found</p>}
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <FileText size={16} className="text-vigil-purple" />
              PDF Reports
            </h3>
            <div className="flex gap-1">
              <a href={api.generateReport('daily')} target="_blank" rel="noreferrer" className="btn-ghost text-[10px]">Daily</a>
              <a href={api.generateReport('weekly')} target="_blank" rel="noreferrer" className="btn-ghost text-[10px]">Weekly</a>
              <a href={api.generateReport('monthly')} target="_blank" rel="noreferrer" className="btn-ghost text-[10px]">Monthly</a>
            </div>
          </div>
          <div className="space-y-1.5 max-h-[calc(100vh-240px)] overflow-y-auto">
            {reports.map((rep) => (
              <div key={rep.filename} className="flex items-center justify-between p-2.5 rounded-vigil-xs bg-vigil-surface2">
                <div>
                  <p className="text-xs font-medium">{rep.filename}</p>
                  <p className="text-[10px] text-vigil-dim font-mono">{(rep.size / 1024).toFixed(1)} KB · {new Date(rep.created).toLocaleString()}</p>
                </div>
                <a href={rep.download_url} download className="btn-icon w-7 h-7" title="Download">
                  <Download size={13} />
                </a>
              </div>
            ))}
            {reports.length === 0 && <p className="text-xs text-vigil-muted text-center py-8">No reports generated yet</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
