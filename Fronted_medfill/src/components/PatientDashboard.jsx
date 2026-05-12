/**
 * src/components/PatientDashboard.jsx — Premium redesign
 * Shows structured patient info + lab panels with health score
 */

import {
  User, Calendar, Building2, Activity, FlaskConical,
  AlertTriangle, ChevronDown, ChevronUp, Heart, Droplets,
  TrendingUp, TrendingDown, Minus, ShieldCheck,
} from 'lucide-react'
import { useState, useMemo } from 'react'

/* ── Helpers ────────────────────────────────────────────────────────────────── */
function fmt(val, fallback = '—') {
  if (val === null || val === undefined || val === '') return fallback
  return String(val)
}

function flagBadge(flag) {
  const map = {
    high:     { cls: 'badge-high',    label: 'HIGH'     },
    low:      { cls: 'badge-low',     label: 'LOW'      },
    critical: { cls: 'badge-high',    label: 'CRITICAL' },
    normal:   { cls: 'badge-normal',  label: 'NORMAL'   },
    pending:  { cls: 'badge-pending', label: 'PENDING'  },
    unknown:  { cls: 'badge-unknown', label: '—'        },
  }
  const { cls, label } = map[flag?.toLowerCase()] ?? map.unknown
  return <span className={`badge ${cls}`}>{label}</span>
}

function isAbnormal(flag) {
  return ['high', 'low', 'critical'].includes(flag?.toLowerCase())
}

function TrendIcon({ flag }) {
  if (flag === 'high')   return <TrendingUp   className="w-3.5 h-3.5 text-red-500"   />
  if (flag === 'low')    return <TrendingDown  className="w-3.5 h-3.5 text-amber-500" />
  if (flag === 'normal') return <Minus         className="w-3.5 h-3.5 text-emerald-500" />
  return null
}

/* ── Health Score ───────────────────────────────────────────────────────────── */
function HealthScore({ panels }) {
  const score = useMemo(() => {
    const rows = panels?.flatMap(p => p.results ?? []) ?? []
    if (!rows.length) return null
    const normal = rows.filter(r => r.flag === 'normal').length
    return Math.round((normal / rows.length) * 100)
  }, [panels])

  if (score === null) return null

  const color = score >= 80 ? '#10b981' : score >= 60 ? '#f59e0b' : '#ef4444'
  const label = score >= 80 ? 'Good'    : score >= 60 ? 'Fair'    : 'Abnormal'
  const r = 36, circ = 2 * Math.PI * r
  const dash = circ * (score / 100)

  return (
    <div className="glass-card p-4 flex items-center gap-4 animate-scale-in">
      <div className="relative w-20 h-20 flex-shrink-0">
        <svg className="rotate-[-90deg]" width="80" height="80">
          <circle cx="40" cy="40" r={r} fill="none" strokeWidth="8" stroke="#f1f5f9" />
          <circle
            cx="40" cy="40" r={r} fill="none" strokeWidth="8"
            stroke={color} strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            style={{ transition: 'stroke-dasharray 1s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold text-slate-800">{score}</span>
          <span className="text-[9px] font-semibold text-slate-400 uppercase tracking-wide">score</span>
        </div>
      </div>
      <div>
        <p className="text-xs text-slate-400 uppercase tracking-widest font-bold mb-0.5">Health Score</p>
        <p className="text-2xl font-extrabold" style={{ color }}>{label}</p>
        <p className="text-xs text-slate-500 mt-1">
          {panels?.flatMap(p => p.results ?? []).filter(r => r.flag === 'normal').length}
          /{panels?.flatMap(p => p.results ?? []).length} values in normal range
        </p>
      </div>
    </div>
  )
}

/* ── Info card ──────────────────────────────────────────────────────────────── */
function InfoCard({ label, value, icon: Icon }) {
  if (!value || value === '—') return null
  return (
    <div className="flex items-start gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100">
      {Icon && (
        <div className="w-7 h-7 rounded-lg bg-teal-50 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Icon className="w-3.5 h-3.5 text-teal-600" />
        </div>
      )}
      <div className="min-w-0">
        <p className="section-title mb-0.5">{label}</p>
        <p className="font-semibold text-slate-800 text-sm truncate">{value}</p>
      </div>
    </div>
  )
}

/* ── Section block ──────────────────────────────────────────────────────────── */
function SectionBlock({ icon: Icon, title, badge, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="glass-card overflow-hidden animate-slide-up">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4
                   hover:bg-slate-50/70 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-teal-50 to-cyan-50
                          border border-teal-100 flex items-center justify-center">
            <Icon className="w-4 h-4 text-teal-600" />
          </div>
          <span className="font-semibold text-slate-700">{title}</span>
          {badge && (
            <span className="ml-1 px-2 py-0.5 rounded-full text-xs font-bold bg-red-100 text-red-600">
              {badge}
            </span>
          )}
        </div>
        {open
          ? <ChevronUp   className="w-4 h-4 text-slate-300" />
          : <ChevronDown className="w-4 h-4 text-slate-300" />}
      </button>
      {open && (
        <div className="px-5 pb-5 border-t border-slate-100 animate-fade-in">
          {children}
        </div>
      )}
    </div>
  )
}

/* ── Mini bar for value range ───────────────────────────────────────────────── */
function RangeBar({ value, low, high }) {
  if (!low || !high || value == null) return null
  const pct = Math.min(100, Math.max(0, ((value - low) / (high - low)) * 100))
  const color = value < low ? '#f59e0b' : value > high ? '#ef4444' : '#10b981'
  return (
    <div className="mt-1.5 h-1 w-full bg-slate-100 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  )
}

/* ── Lab panel table ────────────────────────────────────────────────────────── */
function LabPanel({ panel }) {
  const abnormalCount = panel.results?.filter(r => isAbnormal(r.flag)).length ?? 0
  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-bold text-slate-600 flex items-center gap-2">
          <Droplets className="w-3.5 h-3.5 text-teal-500" />
          {panel.panel_name}
        </h4>
        {abnormalCount > 0 && (
          <span className="flex items-center gap-1 text-xs text-red-600 font-semibold bg-red-50 px-2 py-0.5 rounded-full border border-red-100">
            <AlertTriangle className="w-3 h-3" />
            {abnormalCount} abnormal
          </span>
        )}
      </div>

      <div className="rounded-xl border border-slate-200 overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-[1fr_auto_auto_auto] gap-3 px-4 py-2.5
                        bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
          {['Test', 'Value', 'Range', 'Status'].map(h => (
            <span key={h} className="section-title">{h}</span>
          ))}
        </div>

        {/* Rows */}
        {panel.results?.map((r, i) => {
          const abnorm = isAbnormal(r.flag)
          return (
            <div
              key={i}
              className={`grid grid-cols-[1fr_auto_auto_auto] gap-3 items-center
                px-4 py-3 transition-colors
                ${abnorm
                  ? 'bg-red-50/60 border-l-[3px] border-red-400 hover:bg-red-50'
                  : i % 2 === 0 ? 'hover:bg-slate-50/70' : 'bg-slate-50/30 hover:bg-slate-50/70'
                }`}
            >
              <div>
                <span className={`text-sm font-medium ${abnorm ? 'text-red-800' : 'text-slate-700'}`}>
                  {r.test_name}
                </span>
                <RangeBar
                  value={r.value}
                  low={r.reference_low}
                  high={r.reference_high}
                />
              </div>

              <span className={`text-sm font-bold tabular-nums text-right
                ${abnorm ? 'text-red-700' : 'text-slate-800'}`}>
                {r.value ?? r.value_text ?? '—'}
                {r.unit && <span className="text-[10px] text-slate-400 ml-1 font-normal">{r.unit}</span>}
              </span>

              <span className="text-xs text-slate-400 text-right whitespace-nowrap">
                {r.reference_text ?? (r.reference_low != null
                  ? `${r.reference_low}–${r.reference_high}`
                  : '—')}
              </span>

              <div className="flex items-center gap-1.5 justify-end">
                <TrendIcon flag={r.flag} />
                {flagBadge(r.flag)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── Main Component ─────────────────────────────────────────────────────────── */
export default function PatientDashboard({ payload }) {
  if (!payload) return null

  const {
    patient, facility, vitals, lab_panels, diagnoses,
    medications, clinical_history, clinical_summary,
    report_date, ocr_confidence, elapsed_s,
  } = payload

  const totalAbnormal = lab_panels?.flatMap(p => p.results ?? [])
    .filter(r => isAbnormal(r.flag)).length ?? 0

  return (
    <div className="space-y-4 animate-fade-in">

      {/* ── Abnormal alert banner ── */}
      {totalAbnormal > 0 && (
        <div className="flex items-center gap-3 px-4 py-3 bg-red-50 border border-red-200
                        rounded-2xl animate-scale-in shadow-sm shadow-red-100">
          <div className="w-8 h-8 rounded-xl bg-red-100 flex items-center justify-center flex-shrink-0">
            <AlertTriangle className="w-4 h-4 text-red-500" />
          </div>
          <div>
            <p className="text-sm text-red-700 font-bold">
              {totalAbnormal} abnormal value{totalAbnormal !== 1 ? 's' : ''} detected
            </p>
            <p className="text-xs text-red-500">Flagged rows are highlighted with a red border below</p>
          </div>
        </div>
      )}

      {/* ── Health Score ── */}
      {lab_panels?.length > 0 && <HealthScore panels={lab_panels} />}

      {/* ── Meta strip ── */}
      <div className="flex flex-wrap gap-2">
        {report_date      && <span className="pill"><Calendar className="w-3 h-3" /> {report_date}</span>}
        {ocr_confidence   && <span className="pill"><ShieldCheck className="w-3 h-3" /> OCR {Math.round(ocr_confidence*100)}% confidence</span>}
        {elapsed_s        && <span className="pill"><Activity className="w-3 h-3" /> {elapsed_s}s</span>}
      </div>

      {/* ── Patient Info ── */}
      <SectionBlock icon={User} title="Patient Information">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4">
          <InfoCard label="Full Name"    value={fmt(patient?.name)}           icon={User}     />
          <InfoCard label="Age"          value={patient?.age_years ? `${patient.age_years} yrs` : null} icon={Calendar} />
          <InfoCard label="Sex"          value={fmt(patient?.sex)}            icon={Heart}    />
          <InfoCard label="Patient ID"   value={fmt(patient?.patient_id)}     icon={Activity} />
          <InfoCard label="Contact"      value={fmt(patient?.contact_number)} icon={Activity} />
          <InfoCard label="Address"      value={fmt(patient?.address)}        icon={Building2}/>
        </div>
      </SectionBlock>

      {/* ── Facility ── */}
      {facility?.name && (
        <SectionBlock icon={Building2} title="Facility" defaultOpen={false}>
          <div className="grid grid-cols-2 gap-3 mt-4">
            <InfoCard label="Lab / Hospital" value={fmt(facility?.name)}    icon={Building2} />
            <InfoCard label="Address"        value={fmt(facility?.address)} icon={Building2} />
          </div>
        </SectionBlock>
      )}

      {/* ── Vitals ── */}
      {vitals && Object.values(vitals).some(v => v != null) && (
        <SectionBlock icon={Activity} title="Vitals">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
            {[
              { label: 'Systolic BP',   val: vitals.blood_pressure_systolic,  unit: 'mmHg', warn: vitals.blood_pressure_systolic > 140  },
              { label: 'Diastolic BP',  val: vitals.blood_pressure_diastolic, unit: 'mmHg', warn: vitals.blood_pressure_diastolic > 90   },
              { label: 'Heart Rate',    val: vitals.heart_rate_bpm,           unit: 'bpm',  warn: vitals.heart_rate_bpm > 100            },
              { label: 'SpO₂',          val: vitals.spo2_percent,             unit: '%',    warn: vitals.spo2_percent < 94               },
              { label: 'Temperature',   val: vitals.temperature_celsius,      unit: '°C',   warn: vitals.temperature_celsius > 38        },
              { label: 'Weight',        val: vitals.weight_kg,                unit: 'kg',   warn: false },
            ].filter(v => v.val != null).map(({ label, val, unit, warn }) => (
              <div key={label} className={`glass-card p-3 ${warn ? 'border-l-[3px] border-amber-400' : ''}`}>
                <p className="section-title mb-1">{label}</p>
                <p className={`text-xl font-bold ${warn ? 'text-amber-600' : 'text-slate-800'}`}>
                  {val}
                  <span className="text-xs font-normal text-slate-400 ml-1">{unit}</span>
                </p>
              </div>
            ))}
          </div>
        </SectionBlock>
      )}

      {/* ── Lab Panels ── */}
      {lab_panels?.length > 0 && (
        <SectionBlock
          icon={FlaskConical}
          title={`Lab Results — ${lab_panels.length} panel${lab_panels.length > 1 ? 's' : ''}`}
          badge={totalAbnormal > 0 ? `${totalAbnormal} abnormal` : null}
        >
          {lab_panels.map((panel, i) => <LabPanel key={i} panel={panel} />)}
        </SectionBlock>
      )}

      {/* ── Clinical Summary ── */}
      {clinical_summary && (
        <div className="glass-card px-5 py-4 animate-slide-up">
          <p className="section-title mb-2">Clinical Summary</p>
          <p className="text-sm text-slate-600 leading-relaxed">{clinical_summary}</p>
        </div>
      )}
    </div>
  )
}
