'use client'

import { useState, useRef, useCallback } from 'react'

interface Props {
  onUpload:     (file: File)   => void
  onSubmitText: (text: string, title: string) => void
}

const ACCEPTED = ['.pdf', '.txt', '.docx', '.md']

type Tab = 'file' | 'text'

export default function UploadZone({ onUpload, onSubmitText }: Props) {
  const [tab,      setTab]      = useState<Tab>('file')
  const [dragging, setDragging] = useState(false)
  const [selected, setSelected] = useState<File | null>(null)
  const [text,     setText]     = useState('')
  const [title,    setTitle]    = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File) => setSelected(file), [])
  const onDrop      = (e: React.DragEvent) => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]) }
  const onDragOver  = (e: React.DragEvent) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { if (e.target.files?.[0]) handleFile(e.target.files[0]) }

  return (
    <div className="space-y-3">
      {/* Tab switcher */}
      <div className="flex rounded-md border border-zinc-800 overflow-hidden">
        {(['file', 'text'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`
              flex-1 py-2 text-sm font-medium transition-colors
              ${tab === t
                ? 'bg-[#111113] text-zinc-200'
                : 'bg-transparent text-zinc-600 hover:text-zinc-400'
              }
            `}
          >
            {t === 'file' ? '📄  Upload file' : '✏️  Paste text'}
          </button>
        ))}
      </div>

      {/* ── File tab ─────────────────────────────────────────────────────── */}
      {tab === 'file' && (
        <>
          <div
            onClick={() => inputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            className={`
              relative cursor-pointer rounded-lg border p-10
              flex flex-col items-center justify-center gap-3 text-center
              transition-colors duration-150
              ${dragging
                ? 'border-[#5e6ad2]/60 bg-[#5e6ad2]/5'
                : selected
                  ? 'border-zinc-700 bg-[#111113]'
                  : 'border-zinc-800 bg-[#111113] hover:border-zinc-700'
              }
            `}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(',')}
              className="hidden"
              onChange={onFileChange}
            />

            {selected ? (
              <div className="flex items-center gap-3">
                <FileIcon />
                <div className="text-left">
                  <p className="text-sm font-medium text-zinc-200">{selected.name}</p>
                  <p className="text-xs text-zinc-600 mt-0.5">
                    {(selected.size / 1024).toFixed(0)} KB · click to change
                  </p>
                </div>
              </div>
            ) : (
              <>
                <UploadIcon dragging={dragging} />
                <div>
                  <p className="text-sm text-zinc-400">
                    {dragging ? 'Drop to upload' : 'Drop document or click to browse'}
                  </p>
                  <p className="text-xs text-zinc-600 mt-1">
                    {ACCEPTED.join('  ·  ')}
                  </p>
                </div>
              </>
            )}
          </div>

          {selected && (
            <div className="flex justify-end">
              <button
                onClick={() => onUpload(selected)}
                className="rounded-md bg-[#5e6ad2] px-4 py-2 text-sm font-medium text-white hover:bg-[#6b78e5] transition-colors"
              >
                Run analysis
              </button>
            </div>
          )}
        </>
      )}

      {/* ── Text tab ─────────────────────────────────────────────────────── */}
      {tab === 'text' && (
        <div className="space-y-3">
          <input
            type="text"
            value={title}
            onChange={e => setTitle(e.target.value)}
            placeholder="Use case title (optional)"
            className="
              w-full rounded-md border border-zinc-800 bg-[#111113]
              px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600
              focus:border-zinc-600 focus:outline-none transition-colors
            "
          />
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder={`Describe the AI system you want to assess…\n\nExample: "We're building a system that uses an LSTM model to predict equipment failures in industrial machinery. It ingests sensor data (temperature, vibration) from IoT devices and outputs a failure probability score and recommended maintenance window."`}
            rows={10}
            className="
              w-full rounded-lg border border-zinc-800 bg-[#111113]
              px-4 py-3 text-sm text-zinc-300 placeholder-zinc-600
              focus:border-zinc-600 focus:outline-none resize-y
              leading-relaxed transition-colors
            "
          />
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-700">{text.length.toLocaleString()} characters</span>
            <button
              onClick={() => text.trim() && onSubmitText(text, title || 'Use case description')}
              disabled={!text.trim()}
              className="
                rounded-md bg-[#5e6ad2] px-4 py-2 text-sm font-medium text-white
                hover:bg-[#6b78e5] transition-colors
                disabled:opacity-40 disabled:cursor-not-allowed
              "
            >
              Run analysis
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function UploadIcon({ dragging }: { dragging: boolean }) {
  return (
    <svg
      width="24" height="24" viewBox="0 0 24 24" fill="none"
      className={`transition-colors ${dragging ? 'text-[#5e6ad2]' : 'text-zinc-600'}`}
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function FileIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-zinc-500 shrink-0"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  )
}
