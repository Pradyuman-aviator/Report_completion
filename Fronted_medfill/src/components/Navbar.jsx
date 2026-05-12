/**
 * src/components/Navbar.jsx — Premium redesign
 */

import { useEffect, useState } from 'react'
import { HeartPulse, Wifi, WifiOff, Activity, Zap } from 'lucide-react'

export default function Navbar() {
  const [online, setOnline] = useState(null)

  useEffect(() => {
    let mounted = true
    const ping = async () => {
      const { checkHealth } = await import('../api.js')
      const ok = await checkHealth()
      if (mounted) setOnline(ok)
    }
    ping()
    const id = setInterval(ping, 15_000)
    return () => { mounted = false; clearInterval(id) }
  }, [])

  return (
    <header className="sticky top-0 z-50 w-full">
      {/* Frosted glass bar */}
      <div className="bg-white/75 backdrop-blur-xl border-b border-slate-200/50 shadow-sm shadow-slate-200/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">

          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative w-8 h-8">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-teal-500 via-cyan-500 to-teal-600
                              flex items-center justify-center shadow-lg shadow-teal-500/30">
                <HeartPulse className="w-4 h-4 text-white" />
              </div>
              {/* Subtle glow pulse */}
              <div className="absolute inset-0 rounded-xl bg-teal-400/20 animate-ping" />
            </div>
            <div className="leading-tight">
              <span className="font-bold text-lg tracking-tight gradient-text">MedFill</span>
              <span className="hidden sm:inline text-slate-400 text-xs ml-2 font-medium">
                Clinical AI Dashboard
              </span>
            </div>
          </div>

          {/* Nav links */}
          <nav className="hidden md:flex items-center gap-1 text-sm font-medium">
            {[
              { href: '#upload',     label: 'Upload' },
              { href: '#dashboard',  label: 'Patient Data' },
              { href: '#prediction', label: 'Analysis' },
            ].map(({ href, label }) => (
              <a
                key={href}
                href={href}
                className="px-3 py-1.5 rounded-lg text-slate-500 hover:text-teal-700
                           hover:bg-teal-50 transition-all duration-150"
              >
                {label}
              </a>
            ))}
          </nav>

          {/* Right side */}
          <div className="flex items-center gap-3">

            {/* Model badge */}
            <span className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg
                             bg-violet-50 border border-violet-200 text-violet-700 text-xs font-semibold">
              <Zap className="w-3 h-3" />
              R² 99.7%
            </span>

            {/* Backend status */}
            <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold
                             border transition-all duration-500
                             ${online === null
                               ? 'bg-slate-50 border-slate-200 text-slate-400'
                               : online
                                 ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                                 : 'bg-red-50 border-red-200 text-red-600'}`}>

              {online === null
                ? <Activity className="w-3 h-3 animate-pulse" />
                : online
                  ? <Wifi className="w-3 h-3" />
                  : <WifiOff className="w-3 h-3" />}

              <span>
                {online === null ? 'Checking…' : online ? 'Backend online' : 'Offline'}
              </span>

              {online && (
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
