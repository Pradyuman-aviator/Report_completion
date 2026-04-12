/**
 * src/components/FileUpload.jsx — Premium redesign
 */

import { useState, useRef, useCallback } from 'react'
import {
  UploadCloud, FileImage, X, CheckCircle2, Loader2,
  ScanLine, Cpu, FlaskConical,
} from 'lucide-react'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']
const MAX_MB   = 20

/* Loading steps shown while backend processes */
const STEPS = [
  { icon: ScanLine,     text: 'Running OCR…',             delay: 0     },
  { icon: Cpu,          text: 'Extracting biomarkers…',   delay: 4000  },
  { icon: FlaskConical, text: 'Running AI imputation…',   delay: 9000  },
]

export default function FileUpload({ onSuccess, onError, isLoading, setIsLoading }) {
  const [dragOver,  setDragOver]  = useState(false)
  const [preview,   setPreview]   = useState(null)
  const [progress,  setProgress]  = useState(0)
  const [stepIdx,   setStepIdx]   = useState(0)
  const [done,      setDone]      = useState(false)
  const inputRef  = useRef(null)
  const timersRef = useRef([])

  const validate = (file) => {
    if (!ACCEPTED.includes(file.type)) return `Unsupported format. Use JPEG, PNG, BMP, TIFF or WEBP.`
    if (file.size > MAX_MB * 1024 * 1024) return `File too large (max ${MAX_MB} MB).`
    return null
  }

  const clearTimers = () => { timersRef.current.forEach(clearTimeout); timersRef.current = [] }

  const handleFile = useCallback(async (file) => {
    const err = validate(file)
    if (err) { onError(err); return }

    clearTimers()
    setDone(false)
    setProgress(0)
    setStepIdx(0)
    setPreview({ name: file.name, size: (file.size / 1024).toFixed(1) + ' KB', url: URL.createObjectURL(file) })

    try {
      setIsLoading(true)

      // Advance step indicators at realistic intervals
      STEPS.forEach(({ delay }, i) => {
        if (i === 0) return
        const t = setTimeout(() => setStepIdx(i), delay)
        timersRef.current.push(t)
      })

      const { uploadScan } = await import('../api.js')
      const payload = await uploadScan(file, (pct) => setProgress(pct))

      setDone(true)
      onSuccess(payload)
    } catch (e) {
      onError(e.message)
      setPreview(null)
    } finally {
      setIsLoading(false)
      clearTimers()
    }
  }, [onSuccess, onError, setIsLoading])

  const onDrop       = (e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }
  const onDragOver   = (e) => { e.preventDefault(); setDragOver(true)  }
  const onDragLeave  = ()  => setDragOver(false)
  const onInputChange= (e) => { const f = e.target.files[0]; if (f) handleFile(f); e.target.value = '' }
  const clearFile    = ()  => { setPreview(null); setDone(false); setProgress(0); clearTimers() }

  return (
    <div className="glass-card overflow-hidden animate-fade-in">

      {/* Card header */}
      <div className="flex items-center gap-3 px-5 pt-5 pb-4 border-b border-slate-100">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-600
                        flex items-center justify-center shadow-md shadow-teal-200 flex-shrink-0">
          <UploadCloud className="w-4.5 h-4.5 text-white w-5 h-5" />
        </div>
        <div>
          <h2 className="font-bold text-slate-800">Upload Report Scan</h2>
          <p className="text-xs text-slate-400 mt-0.5">JPEG · PNG · TIFF · BMP · WEBP — max {MAX_MB} MB</p>
        </div>
      </div>

      <div className="p-5 space-y-4">

        {/* ── Drop zone (visible when no file) ── */}
        {!preview && (
          <div
            onClick={() => inputRef.current?.click()}
            onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
            className={`relative flex flex-col items-center justify-center gap-4
              border-2 border-dashed rounded-2xl py-12 px-6 cursor-pointer
              transition-all duration-300 select-none group
              ${dragOver
                ? 'border-teal-400 bg-teal-50 scale-[1.01]'
                : 'border-slate-200 bg-slate-50/50 hover:border-teal-300 hover:bg-teal-50/40'
              }`}
          >
            {/* Icon container */}
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center
              transition-all duration-300 shadow-sm
              ${dragOver
                ? 'bg-teal-500 shadow-teal-400/40 shadow-lg scale-110'
                : 'bg-white group-hover:bg-teal-50 group-hover:scale-105'
              }`}>
              <UploadCloud className={`w-7 h-7 transition-colors duration-300
                ${dragOver ? 'text-white' : 'text-slate-300 group-hover:text-teal-500'}`} />
            </div>

            <div className="text-center">
              <p className={`font-semibold text-sm transition-colors duration-200
                ${dragOver ? 'text-teal-700' : 'text-slate-500 group-hover:text-slate-700'}`}>
                {dragOver ? '📂 Drop to analyze…' : 'Drag & drop your report here'}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                or{' '}
                <span className="text-teal-600 font-semibold underline underline-offset-2 hover:text-teal-700">
                  browse files
                </span>
              </p>
            </div>

            {/* Accepted formats */}
            <div className="flex gap-1.5 flex-wrap justify-center">
              {['JPEG', 'PNG', 'TIFF', 'BMP', 'WEBP'].map(fmt => (
                <span key={fmt} className="px-2 py-0.5 rounded text-[10px] font-bold
                                           bg-white border border-slate-200 text-slate-400">
                  {fmt}
                </span>
              ))}
            </div>

            {/* Pulse ring on drag */}
            {dragOver && (
              <span className="absolute inset-0 rounded-2xl border-2 border-teal-400
                               animate-ping opacity-20 pointer-events-none" />
            )}
          </div>
        )}

        {/* ── Hidden input ── */}
        <input ref={inputRef} type="file" accept={ACCEPTED.join(',')}
               className="hidden" onChange={onInputChange} />

        {/* ── Preview card ── */}
        {preview && (
          <div className="animate-slide-up space-y-3">
            <div className="flex gap-4 p-4 bg-slate-50 rounded-xl border border-slate-200">

              {/* Thumbnail */}
              <div className="w-20 h-20 rounded-xl overflow-hidden border border-slate-200
                              flex-shrink-0 bg-white shadow-sm">
                <img src={preview.url} alt="preview" className="w-full h-full object-cover" />
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="font-semibold text-slate-700 truncate text-sm flex items-center gap-1.5">
                      <FileImage className="w-4 h-4 text-teal-500 flex-shrink-0" />
                      {preview.name}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5">{preview.size}</p>
                  </div>
                  {!isLoading && (
                    <button onClick={clearFile}
                      className="p-1.5 rounded-lg hover:bg-red-50 text-slate-300
                                 hover:text-red-400 transition-colors flex-shrink-0">
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Loading step indicator */}
                {isLoading && (
                  <div className="mt-3 space-y-1.5">
                    {STEPS.map(({ icon: Icon, text }, i) => (
                      <div key={i} className={`flex items-center gap-2 text-xs transition-all duration-500
                        ${i <= stepIdx ? 'opacity-100' : 'opacity-25'}`}>
                        {i < stepIdx
                          ? <CheckCircle2 className="w-3.5 h-3.5 text-teal-500 flex-shrink-0" />
                          : i === stepIdx
                            ? <Loader2 className="w-3.5 h-3.5 text-teal-500 animate-spin flex-shrink-0" />
                            : <Icon className="w-3.5 h-3.5 text-slate-300 flex-shrink-0" />
                        }
                        <span className={i === stepIdx ? 'text-teal-700 font-semibold' : 'text-slate-400'}>
                          {text}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Success */}
                {done && !isLoading && (
                  <div className="mt-2 flex items-center gap-1.5 text-emerald-600">
                    <CheckCircle2 className="w-4 h-4" />
                    <span className="text-xs font-bold">Analysis complete</span>
                  </div>
                )}
              </div>
            </div>

            {/* Progress bar */}
            {isLoading && (
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500 ease-out
                             bg-gradient-to-r from-teal-500 to-cyan-400"
                  style={{ width: `${Math.max(progress, stepIdx * 35 + 5)}%` }}
                />
              </div>
            )}

            {/* Re-upload */}
            {!isLoading && (
              <button onClick={clearFile}
                className="text-xs text-slate-400 hover:text-teal-600 transition-colors
                           underline underline-offset-2">
                Upload a different file
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
