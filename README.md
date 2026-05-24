# Prism — EU AI Act Compliance Engine

Upload an AI system description. Get a structured, evidence-grounded 12-section compliance report.

---

## Four-agent pipeline

Each agent has a single responsibility. No agent knows about the others — they communicate only through typed signals and read/write to isolated memory views.

| Agent | Responsibility |
|---|---|
| **Extraction** | Parses the document into structured facts: industry, AI capabilities, automation level, affected persons, GPAI components |
| **Analysis** | Six legal dimensions assessed **in parallel** (`ThreadPoolExecutor`) — each independently retrieves, reasons, and produces labelled claims |
| **Validation** | Challenges weak claims against authoritative chunks; parallel for claims with sufficient evidence, sequential for those needing fresh retrieval |
| **Synthesis** | Merges findings into the final 12-section report |

Retrieval is not an agent — it is a **shared service** called by agents through the orchestrator's signal protocol.

---

## Two retrieval layers

Every query fans out across two independent indexes:

**1. User document** — the uploaded file, ingested on submission, chunked and stored in a per-session disk DB (`retrieval/user_dbs/<session>/`). Hybrid BM25 + semantic search, fused with Reciprocal Rank Fusion.

**2. Legal corpus** — 48 hand-authored atomic nodes covering Articles 3, 5, 6, 9–17, 25–26, 50 and Annex I/III of the EU AI Act, stored as a `PageIndexNode` tree and lazy-loaded into memory on first access. Same hybrid retrieval stack.

Agents specify `source_types` in their `RetrievalSignal` to target one or both layers. Pre-computed document vectors mean only the query is embedded at retrieval time.

**Retrieval KV cache** — every retrieval call is keyed by `SHA-256(query + sorted_filters)`. When validation or a retry loop re-issues a query that analysis already fetched, the result is returned from the in-session cache with no embedding call and no I/O. Cache hits are logged as `CACHE_HIT` events in the pipeline event stream.

---

## Memory: segmented read/write

`SessionMemory` is the single shared state object for a pipeline run. Agents never access it directly. Each agent receives a **typed view** that exposes only the fields it legitimately needs:

| View | Can read | Can write |
|---|---|---|
| `ExtractionAgentMemoryView` | document text | `extracted_facts` |
| `AnalysisAgentMemoryView` | facts, retrieved chunks, prior dimensions | per-dimension findings |
| `ValidationAgentMemoryView` | analysis output, chunks | `validation_flags` |
| `SynthesisAgentMemoryView` | all findings, validation output | `final_report` |

Proxy write methods enforce schema at the boundary. A write that violates the schema raises `MemoryWriteError` immediately — agents cannot silently corrupt state that a later stage depends on.

---

## Checkpoint / rollback

The orchestrator saves named `deepcopy` snapshots at four pipeline boundaries:

```
after_extraction → after_analysis → after_validation → after_synthesis
```

When the synthesis agent raises a `LoopSignal` (low confidence, unresolved dimensions), the orchestrator rolls back to `after_analysis`, discards validation and synthesis output, and re-runs from there with the loop context injected. Max one global loop. No agent restart is needed — rollback is a memory pointer swap.

---

## Inter-agent communication: typed signals

Agents return signals, not results directly. The orchestrator owns all routing.

| Signal | Meaning |
|---|---|
| `RetrievalSignal` | "I need more chunks for this dimension/claim" — orchestrator fetches and retries |
| `LoopSignal` | "Confidence too low to finalise" — orchestrator rolls back and re-runs |
| `CompletionSignal` | "Report is ready" — orchestrator finalises and returns |
| `MemoryWriteError` | Schema violation on a proxy write — orchestrator logs and handles |

No agent holds a reference to another agent or to the retrieval layer.

---

## Structured logging

Every pipeline run emits four typed log streams, persisted to SQLite:

| Stream | What it captures |
|---|---|
| `PipelineEvent` | Orchestrator lifecycle: session start/end, agent boundaries, checkpoint saves/restores, retrieval calls, loop triggers |
| `ReasoningEntry` | Every LLM call: prompt hash, response, duration, token counts — captured automatically via decorator |
| `StateChangeEntry` | Every memory proxy write attempt: field name, value hash, success/failure |
| `SignalEntry` | Every inter-agent signal: type, issuing agent, payload |

The logger is injected via `ContextVar` — agent code calls `@log_reasoning` without importing or knowing about the logger.

---

## LLM resilience: detach / reattach

The LLM client wraps every call in an indefinite retry loop. On connection error:
- Clears a `threading.Event` (`_llm_connected`)
- Sleeps 5 s, retries — **forever**
- Job status becomes `waiting_llm`; the UI shows a yellow "paused" banner
- When the LLM returns, the pipeline resumes exactly where it was

Jobs are persisted to disk (`logs/jobs/<id>.json`) on every status change. In-flight jobs survive server restarts.

---

## Sections 11 & 12: programmatic, not LLM-generated

- **§11 Evidence separation** — all claims grouped by epistemological label (`FACT / RETRIEVED / ASSUMPTION / UNCERTAIN`) directly from pipeline state
- **§12 Agent trace** — structured timeline of every stage's inputs and outputs, built from real pipeline data

These sections cannot hallucinate. The LLM produces only the narrative sections (§1–§9) and per-dimension confidence labels (§10).

---

## Stack

- **Backend** — Python, FastAPI, `uvicorn`
- **LLM** — OpenAI-compatible local inference (Qwen 3.6B–27B via vLLM)
- **Embeddings** — `BAAI/bge-m3` (hosted)
- **Retrieval** — Okapi BM25 + cosine semantic, fused with Reciprocal Rank Fusion
- **Frontend** — Next.js 14, Tailwind CSS; full 12-section collapsible report with agent trace timeline

---

## Running locally

```bash
# Backend
pip install -r requirements.txt
uvicorn api.server:app --port 8001 --reload

# Frontend
cd web && npm install && npm run dev   # → http://localhost:3000
```

```
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen3.6-27B
EMBED_BASE_URL=https://...
EMBED_MODEL=BAAI/bge-m3
EMBED_API_KEY=<token>
```
