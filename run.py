"""
run.py
──────
End-to-end Prism compliance pipeline.

Usage:
    python run.py

Steps
─────
  1. Build the legal corpus (PageIndex trees) — idempotent, skipped if already done.
  2. Create the LLM client (any OpenAI-compatible endpoint).
  3. Wire the embedding function (separate endpoint) into the retrieval layer.
  4. Ingest the user PDF into a session-scoped vector store.
  5. Extract raw document text for the extraction agent.
  6. Instantiate the Orchestrator with all hooks wired up.
  7. Run the pipeline and print the final compliance report.

Environment variables (optional overrides)
──────────────────────────────────────────
    LLM_API_KEY    API key for the LLM endpoint   (fallback: LLM_API_KEY env var)
    EMBED_API_KEY  API key for the embed endpoint  (fallback: EMBED_API_KEY env var)
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

USER_DOCUMENT = Path(
    "user_input/mds-predictive-maintenance-modeling-for-industrial-machinery-case-study.pdf"
)
SESSION_ID = "prism_run_001"

# ── LLM (chat/completion) ──────────────────────────────────────────────────────
LLM_BASE_URL          = "http://localhost:8000/v1"        # your local inference server
LLM_MODEL             = "Qwen/Qwen3.6-27B"               # model name as the server knows it
LLM_API_KEY           = os.environ.get("LLM_API_KEY", "none")  # "none"/"EMPTY" for local vLLM
LLM_MAX_OUTPUT_TOKENS = 16384  # output token budget; context window is 131k so this is well within range

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBED_BASE_URL = "https://containers.datacrunch.io/bge-m3/v1"
EMBED_MODEL    = "BAAI/bge-m3"
EMBED_API_KEY  = os.environ.get("EMBED_API_KEY", "none")         # DataCrunch token

RETRIEVAL_DIR = Path("retrieval")
CORPUS_DIR    = Path("retrieval/corpus")


# ══════════════════════════════════════════════════════════════════════════════
# Step helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_corpus_if_needed(llm_client) -> None:
    """Index all *_atomic_nodes.json files into PageIndex trees if not already done.

    Each JSON file gets its own .pageindex.json in CORPUS_DIR.  Already-indexed
    files are skipped so this function is safe to call after adding new corpus
    JSON files — only the new ones will be ingested.
    """
    import json as _json
    import openai
    import time
    from retrieval.ingest import ingest_legal_json

    json_files = sorted(RETRIEVAL_DIR.glob("*_atomic_nodes.json"))
    if not json_files:
        print("[corpus] No *_atomic_nodes.json files found — skipping indexing.")
        return

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    any_ingested = False
    for json_path in json_files:
        out_path = CORPUS_DIR / f"{json_path.stem}.pageindex.json"
        if out_path.exists():
            print(f"[corpus] {json_path.name} → already indexed, skipping.")
            continue

        print(f"[corpus] Indexing {json_path.name} …")
        any_ingested = True

        _raw_nodes   = _json.loads(json_path.read_text(encoding="utf-8"))
        _total_nodes = len(_raw_nodes) + len(_raw_nodes) // 5   # leaves + ~20% containers
        _counter     = [0]

        # Detect source_type from the filename: files with "prohibited" or
        # "guidelines" → official_guidance; all others → legislation.
        stem_lower  = json_path.stem.lower()
        source_type = (
            "official_guidance"
            if any(kw in stem_lower for kw in ("prohibited", "guideline", "guidance"))
            else "legislation"
        )

        def llm_summarise(text: str) -> str:
            _counter[0] += 1
            preview = text[:60].replace("\n", " ")
            print(
                f"[corpus]   node {_counter[0]:>3}/{_total_nodes} — {preview!r:.55s} …",
                flush=True,
            )
            max_retries = 5
            base_delay  = 6.0
            for attempt in range(max_retries):
                try:
                    resp = llm_client.messages.create(
                        model      = LLM_MODEL,
                        max_tokens = 256,
                        system     = (
                            "You are a legal document summariser. "
                            "Produce a single concise sentence (≤ 40 words) summarising "
                            "the key legal requirement or topic of the passage below."
                        ),
                        messages   = [{"role": "user", "content": text[:4000]}],
                    )
                    return resp.content[0].text.strip()
                except openai.RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    wait = base_delay * (2 ** attempt)
                    print(f"[corpus]   Rate limit — retrying in {wait:.0f}s …")
                    time.sleep(wait)
            return text[:300]   # fallback: should not normally be reached

        root = ingest_legal_json(
            json_path,
            CORPUS_DIR,
            source_type = source_type,
            llm_fn      = llm_summarise,
        )
        print(f"[corpus] Saved → {out_path.name}  (root: {root.node_id!r})")

    if not any_ingested:
        n = len(list(CORPUS_DIR.glob("*.pageindex.json")))
        print(f"[corpus] All {n} index file(s) up-to-date.")


def _make_embed_fn(api_key: str | None = None):
    """Return a BGE-M3 embedding function backed by the DataCrunch container."""
    import openai  # noqa: PLC0415

    key = api_key or EMBED_API_KEY
    embed_client = openai.OpenAI(api_key=key, base_url=EMBED_BASE_URL)

    def real_embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
        ordered = sorted(response.data, key=lambda e: e.index)
        return [e.embedding for e in ordered]

    return real_embed


def _ingest_user_document(embed_fn, *, force_reingest: bool = False) -> None:
    """Chunk and embed the user PDF; store in the session vector store."""
    from retrieval.upload_ingest import ingest_files  # noqa: PLC0415

    session_dir = Path(f"retrieval/user_dbs/{SESSION_ID}")
    if session_dir.exists() and not force_reingest:
        print(f"[ingest] Session {SESSION_ID!r} already ingested — skipping.")
        return

    if not USER_DOCUMENT.exists():
        print(f"[ingest] ERROR: user document not found at {USER_DOCUMENT}")
        sys.exit(1)

    print(f"[ingest] Ingesting {USER_DOCUMENT.name} …")
    result = ingest_files(
        file_paths = [USER_DOCUMENT],
        session_id = SESSION_ID,
        embed_fn   = embed_fn,
    )
    print(
        f"[ingest] Done — {result.chunk_count} chunks from "
        f"{result.file_count} file(s) in session {SESSION_ID!r}."
    )


def _extract_document_text() -> str:
    """Read raw text from the user PDF (no chunking, no embedding)."""
    from retrieval.upload_ingest import extract_file_text  # noqa: PLC0415

    print(f"[extract] Extracting text from {USER_DOCUMENT.name} …")
    segments = extract_file_text(USER_DOCUMENT)
    full_text = "\n\n".join(seg["text"] for seg in segments if seg.get("text"))
    print(f"[extract] Extracted {len(full_text):,} characters.")
    return full_text


def _print_report(report) -> None:
    """Pretty-print the final 12-section ReportSection to stdout."""
    if report is None:
        print("\n[error] No compliance report was produced.")
        return

    print("\n" + "═" * 70)
    print("  PRISM COMPLIANCE REPORT  (12 sections)")
    print("═" * 70)

    # ── Section 1: narrative summary ──────────────────────────────────────
    print("\n§1  USE CASE SUMMARY")
    print("─" * 50)
    print(report.use_case_summary or "(not available)")

    # ── Sections 2–10: structured dict sections ───────────────────────────
    dict_sections = [
        ("§2  EXTRACTED FACTS",                 "extracted_facts"),
        ("§3  AI SYSTEM DETERMINATION",          "ai_definition_check"),
        ("§4  RISK CLASSIFICATION",              "risk_classification"),
        ("§5  PROHIBITED-PRACTICE CHECK",        "prohibited_practices_check"),
        ("§6  TRANSPARENCY & LABELING",          "transparency_gpai_obligations"),
        ("§7  PROVIDER / DEPLOYER ROLES",        "roles"),
        ("§8  GOVERNANCE RECOMMENDATIONS",       "governance_observations"),
        ("§9  MISSING INFORMATION / GAPS",       "missing_information"),
        ("§10 CONFIDENCE SCORE",                 "confidence_score"),
        ("    CITATIONS BY SOURCE",              "citations_by_source"),
    ]

    for label, attr in dict_sections:
        value = getattr(report, attr, None)
        print(f"\n{label}")
        print("─" * 50)
        if isinstance(value, dict) and value:
            print(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            print("(not available)")

    # ── Section 11: evidence separation (by epistemological label) ────────
    print("\n§11 EVIDENCE SEPARATION")
    print("─" * 50)
    ev = getattr(report, "evidence_separation", None)
    if ev:
        for label_key in ("RETRIEVED", "FACT", "ASSUMPTION", "UNCERTAIN"):
            claims = ev.get(label_key, [])
            print(f"  {label_key}: {len(claims)} claim(s)")
        print()
        print(json.dumps(ev, indent=2, ensure_ascii=False))
    else:
        print("(not available)")

    # ── Section 12: agent trace ───────────────────────────────────────────
    print("\n§12 AGENT TRACE")
    print("─" * 50)
    trace = getattr(report, "agent_trace", None)
    if trace:
        for stage in trace:
            status_icon = "✓" if stage.get("status") == "completed" else "⟳"
            print(f"  [{status_icon}] Stage {stage.get('stage')}: {stage.get('agent')}")
            print(f"      {stage.get('description', '')}")
        print()
        print(json.dumps(trace, indent=2, ensure_ascii=False))
    else:
        print("(not available)")

    print("\n" + "═" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    from core.llm_client import make_llm_client                     # noqa: PLC0415
    from core.orchestrator import Orchestrator                       # noqa: PLC0415
    from agents.extraction_agent import run_extraction_agent         # noqa: PLC0415
    from retrieval.retriever import retrieve, set_embed_fn           # noqa: PLC0415
    from core.logger import PrismLogger                              # noqa: PLC0415

    # ① Create the LLM client (local OpenAI-compatible server).
    print(f"[llm] Connecting to {LLM_BASE_URL} (model: {LLM_MODEL}) …")
    llm_client = make_llm_client(
        "openai_compat",
        model             = LLM_MODEL,
        api_key           = LLM_API_KEY,
        base_url          = LLM_BASE_URL,
        max_output_tokens = LLM_MAX_OUTPUT_TOKENS,
        disable_thinking  = True,   # Qwen3 puts all tokens in <think> and returns empty content
    )

    # ② Create and register the embedding function (BGE-M3).
    print(f"[embed] Wiring BGE-M3 embeddings from {EMBED_BASE_URL} …")
    embed_fn = _make_embed_fn()
    set_embed_fn(embed_fn)   # used by retriever's semantic search

    # ③ Build the legal corpus index (idempotent).
    _build_corpus_if_needed(llm_client)

    # ④ Ingest the user PDF into the session vector store.
    _ingest_user_document(embed_fn)

    # ⑤ Extract raw document text for the extraction agent.
    document_text = _extract_document_text()

    # ⑥ Wire the Orchestrator with a SQLite logger.
    Path("logs").mkdir(exist_ok=True)
    prism_logger = None
    try:
        prism_logger = PrismLogger(session_id=SESSION_ID, db_path=f"logs/{SESSION_ID}.db")
        print(f"[log] Writing session log → logs/{SESSION_ID}.db")
    except Exception as log_err:
        print(f"[log] WARNING: Could not initialise logger ({log_err}) — continuing without logging.")
        print("[log]          Delete any stale logs/*.db-wal / *.db-shm files if this persists.")

    print("[orchestrator] Wiring pipeline …")
    orchestrator = Orchestrator(
        retrieve_fn   = partial(retrieve, session_id=SESSION_ID),
        extraction_fn = partial(run_extraction_agent, llm_client=llm_client),
        llm_client    = llm_client,
        prism_logger  = prism_logger,   # None → logging silently disabled
        session_id    = SESSION_ID,     # enables augmentation phase
    )

    # ⑦ Run the full compliance analysis pipeline.
    print("[orchestrator] Running pipeline … (this may take a minute)\n")
    try:
        report = orchestrator.run(document_text=document_text)
    finally:
        if prism_logger is not None:
            prism_logger.close()

    # ⑧ Print results.
    _print_report(report)


if __name__ == "__main__":
    main()
