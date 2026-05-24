'use client'

import { JobStatus } from '@/app/page'
import Spinner from '@/components/Spinner'

interface Props { job: JobStatus | null }

const STAGES = [
  { id: 'extracting',   label: 'Extraction',  desc: 'Parsing document structure and facts'     },
  { id: 'analysing',    label: 'Analysis',    desc: 'Mapping provisions against EU AI Act'      },
  { id: 'validating',   label: 'Validation',  desc: 'Cross-checking compliance claims'          },
  { id: 'synthesising', label: 'Synthesis',   desc: 'Generating structured compliance report'   },
] as const

type StageId = typeof STAGES[number]['id'] | 'pending' | 'done' | 'error'

function stageIndex(status: string): number {
  const order = ['extracting', 'analysing', 'validating', 'synthesising']
  const i = order.indexOf(status === 'waiting_llm' ? 'analysing' : status)
  if (status === 'done')    return order.length
  if (status === 'pending') return -1
  return i === -1 ? 0 : i
}

export default function PipelineStatus({ job }: Props) {
  const curr        = stageIndex(job?.status ?? 'pending')
  const isWaiting   = job?.status === 'waiting_llm'

  return (
    <div className="mx-auto max-w-xl">
      <div className="mb-8">
        <h2 className="text-base font-semibold text-zinc-100 mb-1">Analysing document</h2>
        <p className="text-sm text-zinc-500">{job?.stage ?? 'Initialising pipeline…'}</p>
      </div>

      {/* LLM disconnect notice */}
      {isWaiting && (
        <div className="mb-5 flex items-start gap-3 rounded-md border border-yellow-500/20 bg-yellow-500/5 px-4 py-3">
          <span className="mt-0.5 h-2 w-2 rounded-full bg-yellow-400 animate-pulse shrink-0" />
          <div>
            <p className="text-sm font-medium text-yellow-300">LLM unavailable</p>
            <p className="text-xs text-yellow-400/70 mt-0.5">
              The model server is unreachable. The pipeline is paused and will resume
              automatically once the LLM reconnects — no action needed.
            </p>
          </div>
        </div>
      )}

      <div className="space-y-px">
        {STAGES.map((stage, i) => {
          const isDone    = i < curr
          const isActive  = i === curr
          const isPending = i > curr

          return (
            <div
              key={stage.id}
              className={`
                flex items-center gap-4 px-4 py-3 rounded-md transition-colors
                ${isActive ? 'bg-[#111113] border border-zinc-700/60' : ''}
              `}
            >
              {/* Indicator */}
              <div className="shrink-0 w-5 flex justify-center">
                {isDone    && <CheckIcon />}
                {isActive  && !isWaiting && <Spinner size="sm" />}
                {isActive  && isWaiting  && <PauseIcon />}
                {isPending && <DotIcon />}
              </div>

              {/* Text */}
              <div className="flex-1 min-w-0">
                <span className={`text-sm ${
                  isDone    ? 'text-zinc-400 line-through decoration-zinc-700' :
                  isActive  ? 'text-zinc-100 font-medium' :
                              'text-zinc-600'
                }`}>
                  {stage.label}
                </span>
              </div>

              {/* Status */}
              <span className={`text-xs shrink-0 ${
                isDone    ? 'text-zinc-600' :
                isActive && isWaiting ? 'text-yellow-400' :
                isActive  ? 'text-[#5e6ad2]' :
                            'text-zinc-700'
              }`}>
                {isDone ? 'done' : isActive && isWaiting ? 'paused' : isActive ? 'running' : 'waiting'}
              </span>
            </div>
          )
        })}
      </div>

      {/* Progress bar */}
      {job && job.progress > 0 && (
        <div className="mt-6">
          <div className="h-px w-full bg-zinc-800 overflow-hidden rounded-full">
            <div
              className={`h-full transition-all duration-700 ${isWaiting ? 'bg-yellow-500' : 'bg-[#5e6ad2]'}`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-zinc-500">
      <path d="M2.5 7l3 3 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function DotIcon() {
  return <span className="h-1.5 w-1.5 rounded-full bg-zinc-700 inline-block" />
}

function PauseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-yellow-400">
      <rect x="3" y="2.5" width="2.5" height="9" rx="0.5" fill="currentColor" />
      <rect x="8.5" y="2.5" width="2.5" height="9" rx="0.5" fill="currentColor" />
    </svg>
  )
}
