/**
 * src/App.jsx — Premium redesign
 */

import { useState, useRef } from 'react'
import {
  AlertCircle, X, FileText, Zap, RotateCcw,
  Database, Brain, Clock, ShieldCheck,
} from 'lucide-react'

import Navbar           from './components/Navbar.jsx'
import FileUpload       from './components/FileUpload.jsx'
import PatientDashboard from './components/PatientDashboard.jsx'
import PredictionPanel  from './components/PredictionPanel.jsx'

/* ── Step badge ─────────────────────────────────────────────────────────────── */
function StepBadge({ n, label, active, done }) {
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold
      transition-all duration-300
      ${done   ? 'bg-emerald-100 text-emerald-700 border border-emerald-200'
      : active  ? 'bg-gradient-to-r from-teal-600 to-cyan-600 text-white shadow-md shadow-teal-300/40'
      : 'bg-slate-100 text-slate-400'}`}>
      <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold
        ${done ? 'bg-emerald-500 text-white' : active ? 'bg-white/25' : 'bg-slate-200'}`}>
        {done ? '✓' : n}
      </span>
      {label}
    </div>
  )
}

/* ── Error toast ────────────────────────────────────────────────────────────── */
function ErrorToast({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="flex items-start gap-3 px-4 py-3 bg-red-50 border border-red-200
                    rounded-xl animate-slide-up shadow-sm shadow-red-100">
      <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
      <p className="text-sm text-red-700 flex-1">{message}</p>
      <button onClick={onDismiss} className="text-red-400 hover:text-red-600 transition-colors">
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

/* ── Stat pill ──────────────────────────────────────────────────────────────── */
function StatPill({ icon: Icon, label, value, color = 'teal' }) {
  const colors = {
    teal:   'bg-teal-50 border-teal-200 text-teal-700',
    violet: 'bg-violet-50 border-violet-200 text-violet-700',
    amber:  'bg-amber-50 border-amber-200 text-amber-700',
    emerald:'bg-emerald-50 border-emerald-200 text-emerald-700',
  }
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-xs font-semibold ${colors[color]}`}>
      <Icon className="w-3.5 h-3.5" />
      <span className="text-[11px] font-medium opacity-70">{label}</span>
      <span className="font-bold">{value}</span>
    </div>
  )
}

/* ── Empty right column ─────────────────────────────────────────────────────── */
function EmptyDashboard() {
  const features = [
    '🩸 Hemoglobin', '🔬 MCH / MCV / MCHC',
    '🧪 WBC & RBC', '💊 Creatinine',
    '⚡ Sodium & Potassium', '📊 Blood Glucose',
  ]
  return (
    <div className="glass-card flex flex-col items-center justify-center gap-6 py-14 px-8 text-center">

      {/* Icon */}
      <div className="relative">
        <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-teal-500 to-cyan-600
                        flex items-center justify-center shadow-xl shadow-teal-200 animate-float">
          <FileText className="w-9 h-9 text-white" />
        </div>
        <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-violet-500
                        border-2 border-white flex items-center justify-center shadow-md">
          <Brain className="w-3 h-3 text-white" />
        </div>
      </div>

      {/* Copy */}
      <div className="space-y-2">
        <p className="text-lg font-bold text-slate-700">Upload a report to begin</p>
        <p className="text-sm text-slate-400 max-w-xs leading-relaxed">
          Drop any blood test image and AI will extract lab values + predict missing ones instantly
        </p>
      </div>

      {/* What it extracts */}
      <div className="w-full">
        <p className="section-title mb-3">AI extracts & predicts</p>
        <div className="flex flex-wrap gap-2 justify-center">
          {features.map(f => (
            <span key={f} className="px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200
                                      text-xs text-slate-500 font-medium">{f}</span>
          ))}
          <span className="px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200
                           text-xs text-slate-400">+19 more</span>
        </div>
      </div>

      {/* R² badge */}
      <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-violet-50
                      border border-violet-200">
        <span className="text-2xl font-extrabold text-violet-700">99.7%</span>
        <div className="text-left">
          <p className="text-xs font-bold text-violet-600">R² Accuracy</p>
          <p className="text-[10px] text-violet-400">Validated on 2,231 patients</p>
        </div>
      </div>
    </div>
  )
}


/* ── Shimmer skeleton — only shown while uploading ───────────────────────── */
function SkeletonDashboard() {
  return (
    <div className="space-y-4 animate-fade-in">
      <div className="glass-card p-4">
        <div className="animate-shimmer h-4 w-28 rounded-full mb-3" />
        <div className="animate-shimmer h-20 rounded-xl" />
      </div>
      {[140, 100, 180].map((h, i) => (
        <div key={i} className="glass-card overflow-hidden" style={{ height: h }}>
          <div className="animate-shimmer w-full h-full" />
        </div>
      ))}
    </div>
  )
}

/* ── Main App ───────────────────────────────────────────────────────────────── */
export default function App() {
  const [isLoading,        setIsLoading]        = useState(false)
  const [extractedPayload, setExtractedPayload] = useState(null)
  const [errorMsg,         setErrorMsg]         = useState('')

  const dashboardRef  = useRef(null)

  const handleSuccess = (payload) => {
    setExtractedPayload(payload)
    setErrorMsg('')
    setTimeout(() => dashboardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200)
  }

  const handleReset = () => {
    setExtractedPayload(null)
    setErrorMsg('')
    setIsLoading(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const step = extractedPayload ? 2 : isLoading ? 1 : 0

  return (
    <div className="min-h-screen">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">

        {/* ── Hero ── */}
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-6">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-800">
              Clinical Report{' '}
              <span className="gradient-text">Analyzer</span>
            </h1>
            <p className="text-sm text-slate-400 mt-1.5 max-w-md">
              Upload any blood test report — AI extracts known values and predicts missing ones
              using a Transformer trained on 12,637 real patient records.
            </p>

            {/* Stat pills */}
            <div className="flex flex-wrap gap-2 mt-4">
              <StatPill icon={Brain}      label="R² Score"        value="99.7%"   color="violet"  />
              <StatPill icon={Database}   label="Training records" value="12,637" color="teal"    />
              <StatPill icon={Zap}        label="Features"         value="25"     color="amber"   />
              <StatPill icon={ShieldCheck} label="Privacy"         value="100% local" color="emerald" />
              <StatPill icon={Clock}      label="Time"             value="~15s"   color="teal"    />
            </div>
          </div>

          {/* Stepper */}
          <div className="flex items-center gap-2 flex-wrap flex-shrink-0">
            <StepBadge n="1" label="Upload"    active={step === 0} done={step > 0} />
            <span className="text-slate-200 text-sm">→</span>
            <StepBadge n="2" label="Analyzing" active={step === 1} done={step > 1} />
            <span className="text-slate-200 text-sm">→</span>
            <StepBadge n="3" label="Results"   active={step === 2} done={false}    />
          </div>
        </div>

        {/* ── Two-column grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">

          {/* LEFT — Upload + Analysis */}
          <div className="space-y-4">
            <div id="upload">
              <FileUpload
                onSuccess={handleSuccess}
                onError={setErrorMsg}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
              />
            </div>

            <ErrorToast message={errorMsg} onDismiss={() => setErrorMsg('')} />

            {extractedPayload && (
              <button onClick={handleReset} className="btn-secondary text-sm w-full justify-center">
                <RotateCcw className="w-4 h-4" /> Analyze another report
              </button>
            )}

            <div id="prediction">
              <PredictionPanel extractedPayload={extractedPayload} />
            </div>
          </div>

          {/* RIGHT — Patient Dashboard */}
          <div id="dashboard" ref={dashboardRef}>
            {isLoading && !extractedPayload
              ? <SkeletonDashboard />
              : extractedPayload
                ? <PatientDashboard payload={extractedPayload} />
                : <EmptyDashboard />
            }
          </div>
        </div>

        {/* ── Footer ── */}
        <footer className="pt-6 pb-8 border-t border-slate-100 flex flex-col sm:flex-row
                           items-center justify-between gap-3 text-xs text-slate-400">
          <span className="font-medium">MedFill · Local-first AI · Zero cloud · Patient data stays on-device</span>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5 text-violet-400" />
              Transformer · 796K params
            </span>
            <span className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-teal-400" />
              Anemia + CKD + NHANES
            </span>
          </div>
        </footer>
      </main>
    </div>
  )
}
