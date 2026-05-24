'use client'

import { useState, useRef, useCallback } from 'react'
import UploadZone from '@/components/UploadZone'
import PipelineStatus from '@/components/PipelineStatus'
import ReportView from '@/components/ReportView'
import Spinner from '@/components/Spinner'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ReportData {
  use_case_summary:              string
  extracted_facts:               Record<string, unknown>
  ai_definition_check:           Record<string, unknown>
  risk_classification:           Record<string, unknown>
  prohibited_practices_check:    Record<string, unknown>
  transparency_gpai_obligations: Record<string, unknown>
  roles:                         Record<string, unknown>
  governance_observations:       Record<string, unknown>
  missing_information:           Record<string, unknown>
  citations_by_source:           Record<string, unknown>
}

export interface JobStatus {
  job_id:   string
  status:   'pending' | 'extracting' | 'analysing' | 'done' | 'error'
  stage:    string
  progress: number
  result:   ReportData | null
  error:    string | null
}

type Phase = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Home() {
  const [phase,     setPhase]     = useState<Phase>('idle')
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [errorMsg,  setErrorMsg]  = useState('')
  const pollRef                   = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  const handleUpload = useCallback(async (file: File) => {
    setPhase('uploading')
    setErrorMsg('')
    stopPolling()

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch('/api/analyze', { method: 'POST', body: form })
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)

      const { job_id } = await res.json() as { job_id: string }
      setPhase('processing')

      pollRef.current = setInterval(async () => {
        try {
          const sr  = await fetch(`/api/status/${job_id}`)
          if (!sr.ok) return
          const job = await sr.json() as JobStatus
          setJobStatus(job)
          if (job.status === 'done' && job.result) {
            stopPolling(); setPhase('done')
          } else if (job.status === 'error') {
            stopPolling(); setErrorMsg(job.error ?? 'Unknown error'); setPhase('error')
          }
        } catch { /* swallow poll errors */ }
      }, 2500)
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e))
      setPhase('error')
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleReset = () => {
    stopPolling(); setPhase('idle'); setJobStatus(null); setErrorMsg('')
  }

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-zinc-100 flex flex-col">
      {/* Nav */}
      <header className="flex items-center justify-between border-b border-zinc-800/80 px-6 h-12 shrink-0">
        <div className="flex items-center gap-2.5">
          <PrismMark />
          <span className="text-sm font-semibold tracking-tight text-zinc-200">Prism</span>
        </div>
        <span className="label">EU AI Act Compliance Engine</span>
      </header>

      {/* Main */}
      <main className="flex-1 px-4 py-16">
        {phase === 'idle' && <IdleView onUpload={handleUpload} />}

        {phase === 'uploading' && (
          <Center>
            <Spinner />
            <p className="text-sm text-zinc-500">Uploading…</p>
          </Center>
        )}

        {phase === 'processing' && <PipelineStatus job={jobStatus} />}

        {phase === 'done' && jobStatus?.result && (
          <ReportView report={jobStatus.result} onReset={handleReset} />
        )}

        {phase === 'error' && (
          <Center>
            <div className="card max-w-md w-full p-6">
              <p className="label mb-3">Error</p>
              <p className="text-sm text-zinc-400 font-mono break-all mb-5">{errorMsg}</p>
              <button onClick={handleReset} className="btn-secondary text-sm">
                Try again
              </button>
            </div>
          </Center>
        )}
      </main>
    </div>
  )
}

// ── Idle view ─────────────────────────────────────────────────────────────────

function IdleView({ onUpload }: { onUpload: (f: File) => void }) {
  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-100 mb-2">
          EU AI Act Compliance Analysis
        </h1>
        <p className="text-sm text-zinc-500 leading-relaxed max-w-lg">
          Upload AI system documentation to receive a structured risk
          assessment — risk classification, applicable articles,
          obligations, and remediation steps.
        </p>
      </div>

      <UploadZone onUpload={onUpload} />

      <div className="mt-8 grid grid-cols-3 gap-3">
        {[
          { step: '01', title: 'Upload',    desc: 'Drop any PDF, DOCX, or TXT document' },
          { step: '02', title: 'Analyse',   desc: 'Pipeline runs extraction, retrieval, and analysis' },
          { step: '03', title: 'Report',    desc: 'Receive structured risk assessment' },
        ].map(({ step, title, desc }) => (
          <div key={step} className="card p-4">
            <span className="label block mb-2">{step}</span>
            <p className="text-sm font-medium text-zinc-300 mb-1">{title}</p>
            <p className="text-xs text-zinc-600 leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Shared helpers ────────────────────────────────────────────────────────────

function Center({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3">
      {children}
    </div>
  )
}

function PrismMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <polygon
        points="9,1.5 16.5,15.5 1.5,15.5"
        stroke="#5e6ad2"
        strokeWidth="1.5"
        fill="none"
        strokeLinejoin="round"
      />
    </svg>
  )
}
