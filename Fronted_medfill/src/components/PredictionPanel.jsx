/**
 * src/components/PredictionPanel.jsx
 * =====================================
 * Shows the complete 25-feature Report Analysis — auto-displays
 * as soon as the payload arrives (no extra button click needed).
 */

import {
  FlaskConical, Star, CheckCircle2,
  TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp,
  Activity, HeartPulse,
} from 'lucide-react'
import { useState } from 'react'

/* ── Feature metadata for range checks ─────────────────────────────────────── */
const FEATURE_META = {
  Hemoglobin: { normal: [10.0, 18.0] }, MCH:  { normal: [25.0, 35.0] },
  MCHC:       { normal: [30.0, 38.0] }, MCV:  { normal: [60.0, 100.0] },
  bp:         { normal: [60, 140] },    bgr:  { normal: [70, 200] },
  bu:         { normal: [5, 60] },      sc:   { normal: [0.4, 1.4] },
  sod:        { normal: [135, 145] },   pot:  { normal: [3.5, 5.0] },
  pcv:        { normal: [35, 55] },     wc:   { normal: [4000, 11000] },
  rc:         { normal: [3.5, 5.5] },   age:  { normal: [1, 120] },
}

function rangeStatus(col, val) {
  const meta = FEATURE_META[col]
  if (!meta?.normal || val == null) return 'neutral'
  const [lo, hi] = meta.normal
  if (val < lo) return 'low'
  if (val > hi) return 'high'
  return 'normal'
}

function displayValue(info) {
  const { value, display_value } = info
  if (value == null) return '—'
  if (display_value != null) return display_value
  const num = parseFloat(value)
  return isNaN(num) ? String(value) : num.toFixed(2)
}

/* ── Feature row ─────────────────────────────────────────────────────────────── */
function FeatureRow({ col, info }) {
  const status = rangeStatus(col, info.value)
  const isPred = info.predicted
  const colors = { normal: 'text-emerald-700', high: 'text-red-600', low: 'text-amber-600', neutral: 'text-slate-700' }
  const Icon   = status === 'high' ? TrendingUp : status === 'low' ? TrendingDown : Minus

  return (
    <div className={`grid grid-cols-[2fr_1fr_1fr_auto] items-center gap-2 px-3 py-2.5 rounded-lg transition-colors
      ${isPred ? 'bg-violet-50/60 hover:bg-violet-50' : 'hover:bg-slate-50'}`}>

      <div className="flex items-center gap-2 min-w-0">
        {isPred && <Star className="w-3.5 h-3.5 text-violet-500 flex-shrink-0" fill="currentColor" />}
        <span className={`text-sm font-medium truncate ${isPred ? 'text-violet-700' : 'text-slate-700'}`}>
          {info.label ?? col}
        </span>
      </div>

      <div className={`text-sm font-bold tabular-nums text-right ${colors[status]}`}>
        {displayValue(info)}
        {info.unit && <span className="text-[10px] font-normal text-slate-400 ml-1">{info.unit}</span>}
      </div>

      <div className="flex justify-center">
        {isPred
          ? <span className="badge bg-violet-100 text-violet-700 border-violet-200">predicted</span>
          : <span className="badge badge-normal">extracted</span>}
      </div>

      <Icon className={`w-4 h-4 ${colors[status]}`} />
    </div>
  )
}

/* ── Summary strip ───────────────────────────────────────────────────────────── */
function SummaryStrip({ ocr_confidence, n_extracted, n_predicted }) {
  return (
    <div className="grid grid-cols-3 gap-3 mb-5">
      {[
        { label: 'Accuracy',      value: `${Math.round((ocr_confidence ?? 0) * 100)}%`, color: 'text-teal-700',    bg: 'bg-teal-50' },
        { label: 'Extracted',      value: n_extracted ?? 0,                                color: 'text-emerald-700', bg: 'bg-emerald-50' },
        { label: 'AI Predicted',   value: n_predicted ?? 0,                                color: 'text-violet-700',  bg: 'bg-violet-50' },
      ].map(({ label, value, color, bg }) => (
        <div key={label} className={`stat-card ${bg} border-0`}>
          <span className="section-title">{label}</span>
          <span className={`text-2xl font-bold ${color}`}>{value}</span>
        </div>
      ))}
    </div>
  )
}

/* ── Clinical history card ───────────────────────────────────────────────────── */
function ClinicalHistory({ history }) {
  if (!history) return null
  const items = [
    { label: 'Hypertension',    value: history.hypertension    },
    { label: 'Diabetes',        value: history.diabetes        },
    { label: 'Coronary Artery', value: history.coronary_artery },
  ].filter(i => i.value != null)

  if (!items.length) return null
  return (
    <div className="mb-5 p-4 bg-amber-50 border border-amber-200 rounded-xl">
      <p className="section-title mb-3 flex items-center gap-2">
        <HeartPulse className="w-3.5 h-3.5 text-amber-600" /> Clinical History
      </p>
      <div className="flex flex-wrap gap-3">
        {items.map(({ label, value }) => (
          <div key={label} className="flex items-center gap-2">
            <span className="text-xs text-amber-700 font-medium">{label}:</span>
            <span className={`badge text-xs ${value === 'Yes' ? 'bg-red-100 text-red-700 border-red-200' : 'badge-normal'}`}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Main Component ──────────────────────────────────────────────────────────── */
export default function PredictionPanel({ extractedPayload }) {
  const [showRaw, setShowRaw] = useState(false)

  // Results come directly in the upload response — no second call needed
  const panel   = extractedPayload?.complete_panel ?? null
  const history = extractedPayload?.clinical_history ?? null

  /* Empty state */
  if (!extractedPayload) {
    return (
      <div className="glass-card p-6 animate-fade-in">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-teal-50 flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-teal-600" />
          </div>
          <div>
            <h2 className="font-semibold text-slate-800">Report Analysis</h2>
            <p className="text-xs text-slate-400">25 biomarker features · AI-powered analysis</p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-4 py-3 bg-slate-50 rounded-xl border border-dashed border-slate-200 text-sm text-slate-400">
          <Activity className="w-4 h-4 text-slate-300" />
          Upload a report to see the complete analysis.
        </div>
      </div>
    )
  }

  return (
    <div className="glass-card p-6 animate-fade-in">

      {/* Header */}
      <div className="flex items-center justify-between gap-4 mb-5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-teal-50 flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-teal-600" />
          </div>
          <div>
            <h2 className="font-semibold text-slate-800">Report Analysis</h2>
            <p className="text-xs text-slate-400">AI report analysis · 25 biomarkers</p>
          </div>
        </div>
        <span className="flex items-center gap-1.5 text-emerald-600 text-xs font-semibold">
          <CheckCircle2 className="w-4 h-4" /> Complete
        </span>
      </div>

      {/* Stats */}
      <SummaryStrip
        ocr_confidence={extractedPayload.ocr_confidence}
        n_extracted={extractedPayload.n_extracted}
        n_predicted={extractedPayload.n_predicted}
      />

      {/* Clinical history */}
      <ClinicalHistory history={history} />

      {panel && (
        <>
          {/* Legend */}
          <div className="flex flex-wrap gap-4 mb-3 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <Star className="w-3 h-3 text-violet-500" fill="currentColor" /> AI Predicted
            </span>
            <span className="flex items-center gap-1.5">
              <span className="badge badge-normal">extracted</span> From report
            </span>
            <span className="flex items-center gap-1.5">
              <TrendingUp className="w-3 h-3 text-red-500" /> Above normal
            </span>
            <span className="flex items-center gap-1.5">
              <TrendingDown className="w-3 h-3 text-amber-500" /> Below normal
            </span>
          </div>

          {/* Column headers */}
          <div className="grid grid-cols-[2fr_1fr_1fr_auto] gap-2 px-3 py-2
                          bg-slate-50 rounded-t-xl border border-b-0 border-slate-200">
            {['Biomarker', 'Value', 'Source', ''].map(h => (
              <span key={h} className="section-title">{h}</span>
            ))}
          </div>

          {/* Rows */}
          <div className="border border-slate-200 rounded-b-xl overflow-hidden max-h-[420px] overflow-y-auto scrollbar-thin">
            {Object.entries(panel).map(([col, info]) => (
              <FeatureRow key={col} col={col} info={info} />
            ))}
          </div>

          {/* Raw JSON */}
          <button
            onClick={() => setShowRaw(s => !s)}
            className="mt-4 flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-600 transition-colors"
          >
            {showRaw ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            {showRaw ? 'Hide' : 'Show'} raw JSON
          </button>

          {showRaw && (
            <pre className="mt-2 p-4 bg-slate-900 text-emerald-400 text-xs rounded-xl overflow-auto max-h-64 scrollbar-thin font-mono animate-fade-in">
              {JSON.stringify(extractedPayload, null, 2)}
            </pre>
          )}
        </>
      )}
    </div>
  )
}
