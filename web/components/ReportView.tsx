'use client'

import { ReportData } from '@/app/page'

interface Props { report: ReportData; onReset: () => void }

// ── Risk helpers ──────────────────────────────────────────────────────────────

type Risk = 'unacceptable' | 'high' | 'limited' | 'minimal' | 'unknown'

function detectRisk(rc: Record<string, unknown>): Risk {
  const level = String(rc?.risk_level ?? '').toLowerCase()
  if (level === 'unacceptable')  return 'unacceptable'
  if (level === 'high')          return 'high'
  if (level === 'limited')       return 'limited'
  if (level === 'minimal')       return 'minimal'
  // fallback: scan full JSON
  const s = JSON.stringify(rc).toLowerCase()
  if (s.includes('unacceptable')) return 'unacceptable'
  if (s.includes('high risk'))    return 'high'
  if (s.includes('limited risk')) return 'limited'
  if (s.includes('minimal risk')) return 'minimal'
  return 'unknown'
}

const RISK: Record<Risk, { label: string; dot: string; text: string }> = {
  unacceptable: { label: 'Prohibited',   dot: 'bg-red-500',     text: 'text-red-400'     },
  high:         { label: 'High Risk',    dot: 'bg-orange-500',  text: 'text-orange-400'  },
  limited:      { label: 'Limited Risk', dot: 'bg-yellow-400',  text: 'text-yellow-400'  },
  minimal:      { label: 'Minimal Risk', dot: 'bg-emerald-500', text: 'text-emerald-400' },
  unknown:      { label: 'Unclassified', dot: 'bg-zinc-500',    text: 'text-zinc-400'    },
}

// Convert "HIGH" / "MEDIUM" / "LOW" / "INSUFFICIENT" → 0-100
function confidenceToPercent(conf: unknown): number {
  const s = String(conf ?? '').toUpperCase()
  if (s === 'HIGH')         return 88
  if (s === 'MEDIUM')       return 60
  if (s === 'LOW')          return 35
  if (s === 'INSUFFICIENT') return 0
  // numeric fallback
  const n = parseFloat(s)
  if (!isNaN(n)) return Math.min(100, n <= 1 ? Math.round(n * 100) : Math.round(n))
  return 0
}

function deriveConfidence(report: ReportData): number {
  const rc   = report.risk_classification?.confidence
  const def  = report.ai_definition_check?.confidence
  if (rc)  return confidenceToPercent(rc)
  if (def) return confidenceToPercent(def)
  return 0
}

// Pull a plain string "finding" out of a dict section
function getFinding(section: Record<string, unknown>): string {
  return String(section?.finding ?? section?.summary ?? '')
}

// Pull a list of article refs from a section
function getArticles(section: Record<string, unknown>): string[] {
  const refs = section?.article_references ?? section?.triggered_articles ?? []
  return Array.isArray(refs) ? refs.map(String) : []
}

// Pull string lists from missing_information
function getMissingItems(mi: Record<string, unknown>): string[] {
  const gaps = mi?.gaps ?? mi?.missing_evidence ?? mi?.items ?? []
  if (Array.isArray(gaps)) return gaps.map(String)
  return []
}

// Uncertain claims surface in missing_information.uncertain_claims
function getUncertainClaims(mi: Record<string, unknown>): string[] {
  const claims = mi?.uncertain_claims ?? []
  if (Array.isArray(claims)) return claims.map(String)
  return []
}

// Unresolved dimensions (e.g. ["risk_classification", "transparency"])
function getUnresolvedDimensions(mi: Record<string, unknown>): string[] {
  const dims = mi?.unresolved_dimensions ?? []
  if (Array.isArray(dims)) return dims.map(String)
  return []
}

// Pull obligations from transparency or governance
function getObligations(t: Record<string, unknown>, g: Record<string, unknown>): string[] {
  const tObl = t?.obligations ?? t?.requirements ?? []
  const gObl = g?.requirements ?? g?.obligations ?? g?.observations ?? []
  const all = [...(Array.isArray(tObl) ? tObl : []), ...(Array.isArray(gObl) ? gObl : [])]
  return all.map(String).filter(Boolean)
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ReportView({ report, onReset }: Props) {
  const risk   = detectRisk(report.risk_classification)
  const rc     = RISK[risk]
  const scoreP = deriveConfidence(report)

  const useCaseName = String(
    report.extracted_facts?.use_case_name ?? 'AI System'
  )
  const rcFinding   = getFinding(report.risk_classification)
  const defFinding  = getFinding(report.ai_definition_check)
  const isAiSystem  = report.ai_definition_check?.is_ai_system
  const articles    = getArticles(report.risk_classification)
  const obligations = getObligations(
    report.transparency_gpai_obligations,
    report.governance_observations
  )
  const gaps             = getMissingItems(report.missing_information)
  const uncertainClaims  = getUncertainClaims(report.missing_information)
  const unresolvedDims   = getUnresolvedDimensions(report.missing_information)
  const prohibFinding    = getFinding(report.prohibited_practices_check)
  const prohibited       = report.prohibited_practices_check?.prohibited

  return (
    <div className="mx-auto max-w-2xl">

      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <p className="label mb-1.5">Compliance Report</p>
          <h1 className="text-xl font-semibold text-zinc-100 leading-tight">
            {useCaseName}
          </h1>
        </div>
        <button
          onClick={onReset}
          className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors mt-1"
        >
          ← New analysis
        </button>
      </div>

      {/* Summary cards row */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        {/* Risk */}
        <div className="card p-4">
          <p className="label mb-3">Risk Classification</p>
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full shrink-0 ${rc.dot}`} />
            <span className={`text-sm font-semibold ${rc.text}`}>{rc.label}</span>
          </div>
          {articles.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5 border-t border-zinc-800 pt-3">
              {articles.map((a, i) => (
                <span key={i} className="rounded border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-400">
                  {a}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Confidence */}
        <div className="card p-4">
          <p className="label mb-3">Analysis Confidence</p>
          <p className="text-2xl font-semibold text-zinc-100 tabular-nums">
            {scoreP > 0 ? `${scoreP}%` : '—'}
          </p>
          {scoreP > 0 && (
            <>
              <div className="mt-3 h-px bg-zinc-800 overflow-hidden rounded-full">
                <div
                  className={`h-full transition-all duration-500 ${
                    scoreP >= 75 ? 'bg-emerald-500' : scoreP >= 50 ? 'bg-yellow-400' : 'bg-red-400'
                  }`}
                  style={{ width: `${scoreP}%` }}
                />
              </div>
              <p className="text-xs text-zinc-600 mt-2">
                {scoreP >= 75 ? 'High confidence' : scoreP >= 50 ? 'Moderate — review recommended' : 'Low — manual review required'}
              </p>
            </>
          )}
          {scoreP === 0 && (
            <p className="text-xs text-zinc-600 mt-2">Insufficient evidence for full classification</p>
          )}
        </div>
      </div>

      {/* Summary */}
      {report.use_case_summary && (
        <Section title="Summary">
          <p className="text-sm text-zinc-400 leading-relaxed">{report.use_case_summary}</p>
        </Section>
      )}

      {/* AI System Definition */}
      {defFinding && (
        <Section title="AI System Definition">
          <div className="flex items-center gap-2 mb-3">
            <span className={`h-1.5 w-1.5 rounded-full ${isAiSystem === false ? 'bg-zinc-500' : 'bg-[#5e6ad2]'}`} />
            <span className="text-xs text-zinc-500">
              {isAiSystem === true ? 'Qualifies as AI system under Article 3(1)' :
               isAiSystem === false ? 'Does not qualify as AI system' :
               'Classification pending further review'}
            </span>
          </div>
          <p className="text-sm text-zinc-400 leading-relaxed">{defFinding}</p>
        </Section>
      )}

      {/* Risk Finding */}
      {rcFinding && (
        <Section title="Risk Assessment">
          <p className="text-sm text-zinc-400 leading-relaxed">{rcFinding}</p>
        </Section>
      )}

      {/* Obligations */}
      {obligations.length > 0 && (
        <Section title="Obligations & Requirements">
          <BulletList items={obligations} />
        </Section>
      )}

      {/* Prohibited Practices */}
      {prohibFinding && (
        <Section title="Prohibited Practices Check">
          <div className="flex items-center gap-2 mb-3">
            <span className={`h-1.5 w-1.5 rounded-full ${prohibited === true ? 'bg-red-500' : 'bg-emerald-500'}`} />
            <span className="text-xs text-zinc-500">
              {prohibited === true ? 'One or more prohibited practices may apply'
                : prohibited === false ? 'No prohibited practices identified'
                : 'Determination pending'}
            </span>
          </div>
          <p className="text-sm text-zinc-400 leading-relaxed">{prohibFinding}</p>
        </Section>
      )}

      {/* Obligations */}
      {obligations.length > 0 && (
        <Section title="Obligations & Requirements">
          <BulletList items={obligations} />
        </Section>
      )}

      {/* Gaps */}
      {(gaps.length > 0 || unresolvedDims.length > 0) && (
        <Section title="Information Gaps">
          {unresolvedDims.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-4">
              {unresolvedDims.map((d, i) => (
                <span key={i} className="rounded border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[11px] text-yellow-400/70">
                  {d.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          )}
          <BulletList items={gaps} accent="text-yellow-400/80" />
        </Section>
      )}

      {/* Uncertain claims */}
      {uncertainClaims.length > 0 && (
        <Section title="Uncertain Claims">
          <BulletList items={uncertainClaims} muted />
        </Section>
      )}

    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-5">
      <div className="border-t border-zinc-800 pt-5">
        <p className="label mb-3">{title}</p>
        {children}
      </div>
    </div>
  )
}

function BulletList({ items, accent, muted }: { items: string[]; accent?: string; muted?: boolean }) {
  if (!items?.length) return <Empty />
  return (
    <ul className="space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex gap-3">
          <span className="mt-1.5 h-1 w-1 rounded-full bg-zinc-600 shrink-0" />
          <span className={`text-sm leading-relaxed ${muted ? 'text-zinc-600' : accent ?? 'text-zinc-400'}`}>
            {item}
          </span>
        </li>
      ))}
    </ul>
  )
}

function Empty() {
  return <p className="text-sm text-zinc-700 italic">None identified.</p>
}
