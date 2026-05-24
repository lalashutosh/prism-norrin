'use client'

import { useState } from 'react'
import { ReportData } from '@/app/page'

interface Props { report: ReportData; onReset: () => void }

// ── Risk helpers ──────────────────────────────────────────────────────────────

type Risk = 'unacceptable' | 'high' | 'limited' | 'minimal' | 'unknown'

function detectRisk(rc: Record<string, unknown>): Risk {
  const level = String(rc?.risk_level ?? '').toLowerCase()
  if (level === 'unacceptable') return 'unacceptable'
  if (level === 'high')         return 'high'
  if (level === 'limited')      return 'limited'
  if (level === 'minimal')      return 'minimal'
  const s = JSON.stringify(rc).toLowerCase()
  if (s.includes('unacceptable')) return 'unacceptable'
  if (s.includes('high risk'))    return 'high'
  if (s.includes('limited risk')) return 'limited'
  if (s.includes('minimal risk')) return 'minimal'
  return 'unknown'
}

const RISK_META: Record<Risk, { label: string; dot: string; text: string; bg: string }> = {
  unacceptable: { label: 'Prohibited',   dot: 'bg-red-500',     text: 'text-red-400',     bg: 'border-red-500/20 bg-red-500/5'     },
  high:         { label: 'High Risk',    dot: 'bg-orange-500',  text: 'text-orange-400',  bg: 'border-orange-500/20 bg-orange-500/5' },
  limited:      { label: 'Limited Risk', dot: 'bg-yellow-400',  text: 'text-yellow-400',  bg: 'border-yellow-400/20 bg-yellow-400/5' },
  minimal:      { label: 'Minimal Risk', dot: 'bg-emerald-500', text: 'text-emerald-400', bg: 'border-emerald-500/20 bg-emerald-500/5' },
  unknown:      { label: 'Unclassified', dot: 'bg-zinc-500',    text: 'text-zinc-400',    bg: 'border-zinc-700 bg-zinc-900' },
}

const LABEL_META: Record<string, { text: string; bg: string }> = {
  RETRIEVED:  { text: 'text-[#5e6ad2]',   bg: 'bg-[#5e6ad2]/10 border-[#5e6ad2]/30' },
  FACT:       { text: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30' },
  ASSUMPTION: { text: 'text-yellow-400',  bg: 'bg-yellow-400/10 border-yellow-400/30'  },
  UNCERTAIN:  { text: 'text-zinc-400',    bg: 'bg-zinc-700/30 border-zinc-700'          },
}

const CONF_META: Record<string, { text: string; bar: string }> = {
  HIGH:         { text: 'text-emerald-400', bar: 'bg-emerald-500' },
  MEDIUM:       { text: 'text-yellow-400',  bar: 'bg-yellow-400'  },
  LOW:          { text: 'text-orange-400',  bar: 'bg-orange-500'  },
  INSUFFICIENT: { text: 'text-zinc-500',    bar: 'bg-zinc-600'    },
}

function confPct(c: unknown): number {
  const s = String(c ?? '').toUpperCase()
  return s === 'HIGH' ? 88 : s === 'MEDIUM' ? 60 : s === 'LOW' ? 35 : 0
}

// ── Small helpers ─────────────────────────────────────────────────────────────

const str  = (v: unknown) => String(v ?? '')
const arr  = (v: unknown): string[] => Array.isArray(v) ? v.map(str) : []
const dict = (v: unknown): Record<string, unknown> =>
  (v && typeof v === 'object' && !Array.isArray(v)) ? v as Record<string, unknown> : {}

function LabelBadge({ label }: { label: string }) {
  const m = LABEL_META[label] ?? LABEL_META.UNCERTAIN
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium ${m.text} ${m.bg}`}>
      {label}
    </span>
  )
}

function ConfBadge({ conf }: { conf: string }) {
  const m = CONF_META[conf] ?? CONF_META.INSUFFICIENT
  return (
    <span className={`text-[11px] font-medium ${m.text}`}>{conf}</span>
  )
}

function ArticlePill({ text }: { text: string }) {
  return (
    <span className="rounded border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-400">
      {text}
    </span>
  )
}

// ── Claim list ────────────────────────────────────────────────────────────────

function ClaimList({ claims }: { claims: unknown[] }) {
  const [expanded, setExpanded] = useState(false)
  if (!claims?.length) return null

  const visible = expanded ? claims : claims.slice(0, 3)

  return (
    <div className="mt-3 border-t border-zinc-800 pt-3 space-y-2">
      <p className="text-[11px] text-zinc-600 uppercase tracking-wider">Evidence claims</p>
      {(visible as Record<string, unknown>[]).map((c, i) => (
        <div key={i} className="flex gap-2 items-start">
          <LabelBadge label={str(c.label)} />
          <span className="text-xs text-zinc-500 leading-relaxed flex-1">{str(c.text)}</span>
          <ConfBadge conf={str(c.confidence)} />
        </div>
      ))}
      {claims.length > 3 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors"
        >
          {expanded ? '▲ show less' : `▼ show ${claims.length - 3} more`}
        </button>
      )}
    </div>
  )
}

// ── Section wrapper ───────────────────────────────────────────────────────────

function Section({
  num, title, children, defaultOpen = true,
}: {
  num: string; title: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-t border-zinc-800 pt-5 mb-5">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full text-left mb-3 group"
      >
        <span className="text-[11px] text-zinc-600 font-mono tabular-nums w-5 shrink-0">§{num}</span>
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider flex-1 group-hover:text-zinc-300 transition-colors">
          {title}
        </span>
        <span className="text-zinc-700 text-xs">{open ? '▲' : '▼'}</span>
      </button>
      {open && children}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ReportView({ report, onReset }: Props) {
  const risk = detectRisk(report.risk_classification)
  const rm   = RISK_META[risk]
  const scoreP = confPct(report.risk_classification?.confidence ?? report.ai_definition_check?.confidence)

  const useCaseName = str(
    report.extracted_facts?.use_case_name ?? report.ai_definition_check?.use_case_name ?? 'AI System'
  )

  return (
    <div className="mx-auto max-w-2xl">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <p className="label mb-1.5">Compliance Report</p>
          <h1 className="text-xl font-semibold text-zinc-100 leading-tight">{useCaseName}</h1>
        </div>
        <button onClick={onReset} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors mt-1">
          ← New analysis
        </button>
      </div>

      {/* ── Summary cards ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 mb-8">
        <div className={`card p-4 border ${rm.bg}`}>
          <p className="label mb-3">Risk Classification</p>
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full shrink-0 ${rm.dot}`} />
            <span className={`text-sm font-semibold ${rm.text}`}>{rm.label}</span>
          </div>
          {arr(report.risk_classification?.article_references).length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5 border-t border-zinc-800/50 pt-3">
              {arr(report.risk_classification.article_references).map((a, i) => (
                <ArticlePill key={i} text={a} />
              ))}
            </div>
          )}
        </div>

        <div className="card p-4">
          <p className="label mb-3">Overall Confidence</p>
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
        </div>
      </div>

      {/* ── §1  Use-case summary ─────────────────────────────────────────── */}
      {report.use_case_summary && (
        <Section num="1" title="Use-case summary">
          <p className="text-sm text-zinc-400 leading-relaxed">{report.use_case_summary}</p>
        </Section>
      )}

      {/* ── §2  Extracted facts ──────────────────────────────────────────── */}
      <Section num="2" title="Extracted facts">
        <FactsTable facts={report.extracted_facts} />
      </Section>

      {/* ── §3  AI system determination ─────────────────────────────────── */}
      <Section num="3" title="AI system determination">
        <div className="flex items-center gap-2 mb-3">
          <span className={`h-1.5 w-1.5 rounded-full ${
            report.ai_definition_check?.is_ai_system === false ? 'bg-zinc-500' : 'bg-[#5e6ad2]'
          }`} />
          <span className="text-xs text-zinc-500">
            {report.ai_definition_check?.is_ai_system === true
              ? 'Qualifies as AI system under Article 3(1)'
              : report.ai_definition_check?.is_ai_system === false
              ? 'Does not qualify as AI system'
              : 'Classification pending'}
          </span>
          <ConfBadge conf={str(report.ai_definition_check?.confidence)} />
        </div>
        <p className="text-sm text-zinc-400 leading-relaxed">
          {str(report.ai_definition_check?.finding)}
        </p>
        <ClaimList claims={(report.ai_definition_check?.claims as unknown[]) ?? []} />
      </Section>

      {/* ── §4  Risk classification ──────────────────────────────────────── */}
      <Section num="4" title="Risk classification">
        <div className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 mb-3 ${rm.bg}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${rm.dot}`} />
          <span className={`text-xs font-medium ${rm.text}`}>{rm.label}</span>
          <ConfBadge conf={str(report.risk_classification?.confidence)} />
        </div>
        <p className="text-sm text-zinc-400 leading-relaxed">
          {str(report.risk_classification?.finding)}
        </p>
        <ClaimList claims={(report.risk_classification?.claims as unknown[]) ?? []} />
      </Section>

      {/* ── §5  Prohibited practices ─────────────────────────────────────── */}
      <Section num="5" title="Prohibited-practice check">
        {(() => {
          const prohibited = report.prohibited_practices_check?.prohibited
          return (
            <>
              <div className="flex items-center gap-2 mb-3">
                <span className={`h-1.5 w-1.5 rounded-full ${prohibited === true ? 'bg-red-500' : 'bg-emerald-500'}`} />
                <span className="text-xs text-zinc-500">
                  {prohibited === true ? 'One or more prohibited practices may apply'
                    : prohibited === false ? 'No prohibited practices identified'
                    : 'Determination pending'}
                </span>
                <ConfBadge conf={str(report.prohibited_practices_check?.confidence)} />
              </div>
              {arr(report.prohibited_practices_check?.triggered_articles).length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {arr(report.prohibited_practices_check.triggered_articles).map((a, i) => (
                    <ArticlePill key={i} text={a} />
                  ))}
                </div>
              )}
              <p className="text-sm text-zinc-400 leading-relaxed">
                {str(report.prohibited_practices_check?.finding)}
              </p>
              <ClaimList claims={(report.prohibited_practices_check?.claims as unknown[]) ?? []} />
            </>
          )
        })()}
      </Section>

      {/* ── §6  Transparency & labeling ──────────────────────────────────── */}
      <Section num="6" title="Transparency & labeling obligations">
        <TransparencySection t={report.transparency_gpai_obligations} />
      </Section>

      {/* ── §7  Roles ────────────────────────────────────────────────────── */}
      <Section num="7" title="Provider / deployer roles">
        <RolesSection r={report.roles} />
      </Section>

      {/* ── §8  Governance ───────────────────────────────────────────────── */}
      <Section num="8" title="Governance recommendations">
        <GovernanceSection g={report.governance_observations} />
      </Section>

      {/* ── §9  Missing information ──────────────────────────────────────── */}
      <Section num="9" title="Missing information / gaps" defaultOpen={false}>
        <MissingSection mi={report.missing_information} />
      </Section>

      {/* ── §10 Confidence score ─────────────────────────────────────────── */}
      <Section num="10" title="Confidence score">
        <ConfidenceSection cs={report.confidence_score} />
      </Section>

      {/* ── §11 Evidence separation ──────────────────────────────────────── */}
      <Section num="11" title="Evidence separation" defaultOpen={false}>
        <EvidenceSeparation ev={report.evidence_separation} />
      </Section>

      {/* ── §12 Agent trace ──────────────────────────────────────────────── */}
      <Section num="12" title="Agent trace" defaultOpen={false}>
        <AgentTrace trace={report.agent_trace} />
      </Section>

    </div>
  )
}

// ── §2 Facts table ────────────────────────────────────────────────────────────

function FactsTable({ facts }: { facts: Record<string, unknown> }) {
  const ROWS: [string, string][] = [
    ['description',              'Description'],
    ['industry',                 'Industry / sector'],
    ['end_users',                'End users'],
    ['affected_persons',         'Affected persons'],
    ['data_inputs',              'Data inputs'],
    ['system_outputs',           'Outputs'],
    ['automation_level',         'Automation level'],
    ['human_oversight_mechanism','Human oversight'],
    ['deployment_context',       'Deployment context'],
    ['gpai_components',          'GPAI components'],
    ['ai_capabilities',          'AI capabilities'],
    ['vendor_developer',         'Developer / vendor'],
  ]

  return (
    <div className="divide-y divide-zinc-800">
      {ROWS.map(([key, label]) => {
        const val = facts?.[key]
        if (!val) return null
        const display = Array.isArray(val) ? val.join(', ') : str(val)
        return (
          <div key={key} className="flex gap-4 py-2">
            <span className="text-xs text-zinc-600 w-36 shrink-0 pt-0.5">{label}</span>
            <span className="text-xs text-zinc-300 flex-1">{display}</span>
          </div>
        )
      })}
    </div>
  )
}

// ── §6 Transparency ───────────────────────────────────────────────────────────

function TransparencySection({ t }: { t: Record<string, unknown> }) {
  const applicable    = arr(t?.applicable_obligations)
  const nonApplicable = arr(t?.non_applicable_obligations)
  const finding       = str(t?.finding)
  const conf          = str(t?.confidence)

  return (
    <>
      <div className="flex items-center gap-2 mb-3">
        <ConfBadge conf={conf} />
        {t?.applies_to_gpai === true    && <ArticlePill text="GPAI" />}
        {t?.labelling_required === true  && <ArticlePill text="Labelling required" />}
        {t?.notification_required === true && <ArticlePill text="Notification required" />}
      </div>
      <p className="text-sm text-zinc-400 leading-relaxed mb-4">{finding}</p>

      {applicable.length > 0 && (
        <div className="mb-3">
          <p className="text-[11px] text-emerald-500/70 uppercase tracking-wider mb-2">Applies</p>
          {applicable.map((o, i) => (
            <div key={i} className="flex gap-2 items-start mb-1.5">
              <span className="text-emerald-500 text-xs mt-0.5">✓</span>
              <span className="text-xs text-zinc-400">{o}</span>
            </div>
          ))}
        </div>
      )}
      {nonApplicable.length > 0 && (
        <div>
          <p className="text-[11px] text-zinc-600 uppercase tracking-wider mb-2">Does not apply</p>
          {nonApplicable.map((o, i) => (
            <div key={i} className="flex gap-2 items-start mb-1.5">
              <span className="text-zinc-600 text-xs mt-0.5">✗</span>
              <span className="text-xs text-zinc-600">{o}</span>
            </div>
          ))}
        </div>
      )}
      <ClaimList claims={(t?.claims as unknown[]) ?? []} />
    </>
  )
}

// ── §7 Roles ──────────────────────────────────────────────────────────────────

function RolesSection({ r }: { r: Record<string, unknown> }) {
  const finding = str(r?.finding)
  const conf    = str(r?.confidence)

  const badges = []
  if (r?.is_provider) badges.push({ label: 'Provider', color: 'text-[#5e6ad2] bg-[#5e6ad2]/10 border-[#5e6ad2]/30' })
  if (r?.is_deployer) badges.push({ label: 'Deployer', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' })
  if (r?.is_both)     badges.push({ label: 'Both',     color: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30' })

  return (
    <>
      <div className="flex items-center gap-2 mb-3">
        {badges.map(b => (
          <span key={b.label} className={`rounded border px-2 py-0.5 text-xs font-medium ${b.color}`}>
            {b.label}
          </span>
        ))}
        <ConfBadge conf={conf} />
      </div>
      <p className="text-sm text-zinc-400 leading-relaxed">{finding}</p>
      {arr(r?.article_references).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {arr(r.article_references).map((a, i) => <ArticlePill key={i} text={a} />)}
        </div>
      )}
      <ClaimList claims={(r?.claims as unknown[]) ?? []} />
    </>
  )
}

// ── §8 Governance ─────────────────────────────────────────────────────────────

function GovernanceSection({ g }: { g: Record<string, unknown> }) {
  const finding  = str(g?.finding)
  const conf     = str(g?.confidence)
  const specObl  = dict(g?.specific_obligations)

  const OBL_LABELS: [string, string][] = [
    ['risk_management_system',    'Art. 9 — Risk management'],
    ['data_governance',           'Art. 10 — Data governance'],
    ['technical_documentation',   'Art. 11 — Technical docs'],
    ['record_keeping_logging',    'Art. 12 — Record-keeping'],
    ['transparency_to_deployers', 'Art. 13 — Transparency'],
    ['human_oversight_measures',  'Art. 14 — Human oversight'],
    ['quality_management',        'Arts. 16-17 — Quality mgmt'],
    ['deployer_obligations',      'Art. 26 — Deployer duties'],
    ['drift_and_monitoring',      'Arts. 9(7), 61 — Monitoring'],
    ['incident_reporting',        'Art. 62 — Incident reporting'],
  ]

  return (
    <>
      <div className="flex items-center gap-2 mb-3">
        {g?.documentation_required === true && <ArticlePill text="Docs required" />}
        {g?.oversight_required === true      && <ArticlePill text="Oversight required" />}
        {g?.monitoring_required === true     && <ArticlePill text="Monitoring required" />}
        <ConfBadge conf={conf} />
      </div>
      <p className="text-sm text-zinc-400 leading-relaxed mb-4">{finding}</p>

      {Object.keys(specObl).length > 0 && (
        <div className="border border-zinc-800 rounded-md overflow-hidden">
          <div className="bg-zinc-900/50 px-3 py-2 border-b border-zinc-800">
            <p className="text-[11px] text-zinc-500 uppercase tracking-wider">Article-level obligations</p>
          </div>
          <div className="divide-y divide-zinc-800">
            {OBL_LABELS.map(([key, label]) => {
              const val = specObl[key]
              if (!val) return null
              const s    = str(val)
              const isNA = s.toLowerCase().startsWith('not applicable')
              return (
                <div key={key} className="flex gap-3 px-3 py-2">
                  <span className={`text-[11px] font-medium w-44 shrink-0 pt-0.5 ${isNA ? 'text-zinc-600' : 'text-zinc-400'}`}>
                    {label}
                  </span>
                  <span className={`text-xs flex-1 ${isNA ? 'text-zinc-600 italic' : 'text-zinc-300'}`}>
                    {s}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
      <ClaimList claims={(g?.claims as unknown[]) ?? []} />
    </>
  )
}

// ── §9 Missing information ────────────────────────────────────────────────────

function MissingSection({ mi }: { mi: Record<string, unknown> }) {
  const gaps      = arr(mi?.gaps)
  const uncertain = arr(mi?.uncertain_claims)
  const actions   = arr(mi?.recommended_actions)
  const unresolved = arr(mi?.unresolved_dimensions)

  if (!gaps.length && !uncertain.length && !actions.length) {
    return <p className="text-sm text-zinc-700 italic">No significant gaps identified.</p>
  }

  return (
    <div className="space-y-4">
      {unresolved.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {unresolved.map((d, i) => (
            <span key={i} className="rounded border border-yellow-400/20 bg-yellow-400/5 px-2 py-0.5 text-[11px] text-yellow-400/70">
              {d.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}
      {gaps.length > 0 && (
        <div>
          <p className="text-[11px] text-zinc-600 uppercase tracking-wider mb-2">Gaps</p>
          <ul className="space-y-1.5">
            {gaps.map((g, i) => (
              <li key={i} className="flex gap-2 items-start">
                <span className="text-yellow-400/60 text-xs mt-0.5 shrink-0">!</span>
                <span className="text-xs text-zinc-400">{g}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {actions.length > 0 && (
        <div>
          <p className="text-[11px] text-zinc-600 uppercase tracking-wider mb-2">Recommended actions</p>
          <ul className="space-y-1.5">
            {actions.map((a, i) => (
              <li key={i} className="flex gap-2 items-start">
                <span className="text-[#5e6ad2] text-xs mt-0.5 shrink-0">→</span>
                <span className="text-xs text-zinc-400">{a}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {uncertain.length > 0 && (
        <div>
          <p className="text-[11px] text-zinc-600 uppercase tracking-wider mb-2">Uncertain claims</p>
          <ul className="space-y-1.5">
            {uncertain.map((u, i) => (
              <li key={i} className="flex gap-2 items-start">
                <span className="text-zinc-600 text-xs mt-0.5 shrink-0">?</span>
                <span className="text-xs text-zinc-600">{u}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── §10 Confidence score ──────────────────────────────────────────────────────

function ConfidenceSection({ cs }: { cs: Record<string, unknown> }) {
  const DIMS: [string, string][] = [
    ['definition_check',    'AI definition'],
    ['risk_classification', 'Risk classification'],
    ['prohibited_practices','Prohibited practices'],
    ['transparency',        'Transparency'],
    ['roles',               'Roles'],
    ['governance',          'Governance'],
  ]

  const narrative = str(cs?.narrative)
  const overall   = str(cs?.overall)
  const om        = CONF_META[overall] ?? CONF_META.INSUFFICIENT

  return (
    <div className="space-y-4">
      {/* Per-dimension bars */}
      <div className="space-y-2">
        {DIMS.map(([key, label]) => {
          const conf = str(cs?.[key]).toUpperCase()
          const pct  = confPct(conf)
          const m    = CONF_META[conf] ?? CONF_META.INSUFFICIENT
          return (
            <div key={key} className="flex items-center gap-3">
              <span className="text-xs text-zinc-600 w-36 shrink-0">{label}</span>
              <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${m.bar}`} style={{ width: `${pct}%` }} />
              </div>
              <span className={`text-[11px] font-medium w-16 text-right ${m.text}`}>{conf || '—'}</span>
            </div>
          )
        })}
      </div>

      {/* Overall */}
      <div className={`flex items-center gap-2 rounded-md border px-3 py-2 ${CONF_META[overall]?.bar ? '' : 'border-zinc-800'}`}>
        <span className="text-xs text-zinc-500">Overall:</span>
        <span className={`text-sm font-semibold ${om.text}`}>{overall || '—'}</span>
      </div>

      {narrative && (
        <p className="text-xs text-zinc-500 leading-relaxed border-t border-zinc-800 pt-3">
          {narrative}
        </p>
      )}
    </div>
  )
}

// ── §11 Evidence separation ───────────────────────────────────────────────────

function EvidenceSeparation({ ev }: { ev: Record<string, unknown[]> }) {
  const [activeLabel, setActiveLabel] = useState<string | null>(null)

  const LABELS: [string, string][] = [
    ['RETRIEVED',  'Grounded in legislation / official guidance'],
    ['FACT',       'Stated explicitly in the uploaded document'],
    ['ASSUMPTION', 'Inferred — not directly evidenced'],
    ['UNCERTAIN',  'Could not be resolved after validation'],
  ]

  if (!ev || Object.keys(ev).length === 0) {
    return <p className="text-sm text-zinc-700 italic">Evidence separation not available.</p>
  }

  return (
    <div className="space-y-4">
      {/* Summary pills */}
      <div className="flex flex-wrap gap-2">
        {LABELS.map(([label, desc]) => {
          const count = ev?.[label]?.length ?? 0
          const m     = LABEL_META[label] ?? LABEL_META.UNCERTAIN
          const isActive = activeLabel === label
          return (
            <button
              key={label}
              onClick={() => setActiveLabel(isActive ? null : label)}
              className={`
                flex items-center gap-2 rounded-md border px-3 py-1.5 transition-all
                ${isActive ? `${m.bg} ring-1 ring-current` : 'border-zinc-800 hover:border-zinc-700'}
              `}
            >
              <span className={`text-xs font-medium ${m.text}`}>{label}</span>
              <span className="text-xs text-zinc-600 tabular-nums">{count}</span>
            </button>
          )
        })}
      </div>

      {/* Active label claims */}
      {activeLabel && (ev?.[activeLabel]?.length ?? 0) > 0 && (
        <div className="border border-zinc-800 rounded-md overflow-hidden">
          <div className="bg-zinc-900/40 px-3 py-2 border-b border-zinc-800">
            <p className="text-[11px] text-zinc-500">
              {LABELS.find(([k]) => k === activeLabel)?.[1]}
            </p>
          </div>
          <div className="divide-y divide-zinc-800 max-h-72 overflow-y-auto">
            {(ev[activeLabel] as Record<string, unknown>[]).map((c, i) => (
              <div key={i} className="px-3 py-2">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] text-zinc-600 font-mono">{str(c.dimension).replace(/_/g, ' ')}</span>
                  <ConfBadge conf={str(c.confidence)} />
                </div>
                <p className="text-xs text-zinc-400">{str(c.text)}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── §12 Agent trace ───────────────────────────────────────────────────────────

function AgentTrace({ trace }: { trace: Array<Record<string, unknown>> }) {
  const [expandedStage, setExpandedStage] = useState<number | null>(null)

  if (!trace?.length) {
    return <p className="text-sm text-zinc-700 italic">Agent trace not available.</p>
  }

  return (
    <div className="relative">
      {/* Vertical line */}
      <div className="absolute left-3.5 top-2 bottom-2 w-px bg-zinc-800" />

      <div className="space-y-4">
        {trace.map((stage) => {
          const num       = Number(stage.stage ?? 0)
          const agent     = str(stage.agent)
          const desc      = str(stage.description)
          const status    = str(stage.status)
          const isDone    = status === 'completed'
          const output    = stage.output as Record<string, unknown>
          const isOpen    = expandedStage === num

          return (
            <div key={num} className="relative pl-10">
              {/* Circle */}
              <div className={`
                absolute left-0 w-7 h-7 rounded-full border-2 flex items-center justify-center
                ${isDone ? 'border-zinc-600 bg-zinc-900' : 'border-[#5e6ad2]/50 bg-[#5e6ad2]/10'}
              `}>
                {isDone
                  ? <span className="text-[9px] font-bold text-zinc-500">{num}</span>
                  : <span className="h-1.5 w-1.5 rounded-full bg-[#5e6ad2] animate-pulse" />
                }
              </div>

              {/* Content */}
              <button
                className="w-full text-left"
                onClick={() => setExpandedStage(isOpen ? null : num)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className={`text-sm font-medium ${isDone ? 'text-zinc-300' : 'text-[#5e6ad2]'}`}>
                      {agent}
                    </p>
                    <p className="text-xs text-zinc-600 mt-0.5">{desc}</p>
                  </div>
                  <span className={`text-[10px] shrink-0 mt-0.5 ${isDone ? 'text-zinc-600' : 'text-[#5e6ad2]'}`}>
                    {isDone ? 'done' : 'running'}
                  </span>
                </div>
              </button>

              {/* Stage output summary */}
              {isDone && output && !isOpen && (
                <StageOutputSummary stage={num} output={output} />
              )}

              {/* Expanded JSON */}
              {isOpen && (
                <pre className="mt-2 rounded-md bg-zinc-900 border border-zinc-800 p-3 text-[10px] text-zinc-500 overflow-x-auto max-h-48">
                  {JSON.stringify(output, null, 2)}
                </pre>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function StageOutputSummary({ stage, output }: { stage: number; output: Record<string, unknown> }) {
  if (stage === 1) {
    return (
      <div className="mt-2 flex flex-wrap gap-2">
        {['use_case_name', 'industry', 'deployment_context'].map(k => {
          const v = output[k]
          return v ? (
            <span key={k} className="text-[10px] rounded border border-zinc-800 bg-zinc-900 px-2 py-0.5 text-zinc-500">
              {str(v)}
            </span>
          ) : null
        })}
      </div>
    )
  }
  if (stage === 2) {
    const dims = output.dimensions_covered
    const total = Number(output.total_legal_chunks ?? 0)
    return (
      <p className="mt-1.5 text-[11px] text-zinc-600">
        {total} legal chunks across {Array.isArray(dims) ? dims.length : 0} dimensions
      </p>
    )
  }
  if (stage === 3) {
    const dims = output as Record<string, { confidence: string; claims_count: number } & Record<string, unknown>>
    return (
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1">
        {Object.entries(dims).map(([dim, val]) => {
          if (typeof val !== 'object' || !val) return null
          const conf = str((val as Record<string,unknown>).confidence)
          const m = CONF_META[conf] ?? CONF_META.INSUFFICIENT
          return (
            <div key={dim} className="flex items-center gap-1.5">
              <span className={`h-1.5 w-1.5 rounded-full ${m.bar}`} />
              <span className="text-[10px] text-zinc-600">{dim.replace(/_/g, ' ')}</span>
              <span className={`text-[10px] ${m.text}`}>{conf}</span>
            </div>
          )
        })}
      </div>
    )
  }
  if (stage === 4) {
    const weak      = Number(output.weak_claims_identified ?? 0)
    const confirmed = Number(output.confirmed ?? 0)
    const overturned = Number(output.overturned ?? 0)
    const unresolved = Number(output.unresolved ?? 0)
    return (
      <div className="mt-2 flex gap-4">
        <Stat label="Weak" n={weak} color="text-zinc-500" />
        <Stat label="Confirmed" n={confirmed} color="text-emerald-400" />
        <Stat label="Overturned" n={overturned} color="text-yellow-400" />
        <Stat label="Unresolved" n={unresolved} color="text-orange-400" />
      </div>
    )
  }
  return null
}

function Stat({ label, n, color }: { label: string; n: number; color: string }) {
  return (
    <div className="text-center">
      <p className={`text-base font-semibold tabular-nums ${color}`}>{n}</p>
      <p className="text-[10px] text-zinc-600">{label}</p>
    </div>
  )
}
