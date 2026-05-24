"""
api/server.py
─────────────
FastAPI server exposing the Prism pipeline as a REST API.

Run:
    cd /path/to/prism
    uvicorn api.server:app --port 8001 --reload

Environment variables (same as run.py):
    LLM_BASE_URL          http://localhost:8000/v1
    LLM_MODEL             QuantTrio/Qwen3.6-27B-AWQ-6Bit
    LLM_API_KEY           none
    LLM_MAX_OUTPUT_TOKENS 400
    EMBED_BASE_URL        https://containers.datacrunch.io/bge-m3/v1
    EMBED_MODEL           BAAI/bge-m3
    EMBED_API_KEY         (your DataCrunch token)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Prism API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Job store ─────────────────────────────────────────────────────────────────

class Job(BaseModel):
    job_id:   str
    status:   str = "pending"   # pending | extracting | analysing | done | error
    stage:    str = "Queued"
    progress: int = 0           # 0-100 (approximate)
    result:   Optional[dict] = None
    error:    Optional[str]  = None

_jobs:    dict[str, Job] = {}
_executor = ThreadPoolExecutor(max_workers=2)


# ── Config helpers ────────────────────────────────────────────────────────────

def _cfg(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ── Pipeline runner (runs in thread pool) ─────────────────────────────────────

def _run_pipeline(job_id: str, tmp_path: str) -> None:
    """Execute the full Prism pipeline and update the job record."""
    from functools import partial

    import openai as _openai

    from agents.extraction_agent import run_extraction_agent
    from core.llm_client import make_llm_client
    from core.orchestrator import Orchestrator
    from retrieval.retriever import retrieve, set_embed_fn
    from retrieval.upload_ingest import clear_session, extract_file_text, ingest_files

    job = _jobs[job_id]

    try:
        # ── Configuration ────────────────────────────────────────────────────
        llm_base   = _cfg("LLM_BASE_URL",          "http://localhost:8000/v1")
        llm_model  = _cfg("LLM_MODEL",             "Qwen/Qwen3.6-27B")
        llm_key    = _cfg("LLM_API_KEY",           "none")
        llm_maxtok = int(_cfg("LLM_MAX_OUTPUT_TOKENS", "16384"))
        emb_base   = _cfg("EMBED_BASE_URL",        "https://containers.datacrunch.io/bge-m3/v1")
        emb_model  = _cfg("EMBED_MODEL",           "BAAI/bge-m3")
        emb_key    = _cfg("EMBED_API_KEY",         "none")

        # ── Stage 1: setup ───────────────────────────────────────────────────
        job.status   = "extracting"
        job.stage    = "Creating LLM client…"
        job.progress = 10

        llm_client = make_llm_client(
            "openai_compat",
            model             = llm_model,
            api_key           = llm_key,
            base_url          = llm_base,
            max_output_tokens = llm_maxtok,
            disable_thinking  = True,
        )

        _emb_client = _openai.OpenAI(api_key=emb_key, base_url=emb_base)

        def real_embed(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            resp = _emb_client.embeddings.create(model=emb_model, input=texts)
            return [e.embedding for e in sorted(resp.data, key=lambda e: e.index)]

        set_embed_fn(real_embed)

        # ── Stage 2: ingest ──────────────────────────────────────────────────
        job.stage    = "Ingesting document…"
        job.progress = 25

        session_id = f"web_{job_id[:8]}"
        ingest_files(
            file_paths=[Path(tmp_path)],
            session_id=session_id,
            embed_fn=real_embed,
        )

        # ── Stage 3: extract text ────────────────────────────────────────────
        job.stage    = "Extracting document text…"
        job.progress = 40

        segments      = extract_file_text(Path(tmp_path))
        document_text = "\n\n".join(s["text"] for s in segments if s.get("text"))

        # ── Stage 4: orchestrator ────────────────────────────────────────────
        job.status   = "analysing"
        job.stage    = "Running compliance analysis…"
        job.progress = 55

        orchestrator = Orchestrator(
            retrieve_fn   = partial(retrieve, session_id=session_id),
            extraction_fn = partial(run_extraction_agent, llm_client=llm_client),
            llm_client    = llm_client,
            session_id    = session_id,   # enables augmentation phase
        )

        # Update stage mid-way (synthesis is the last heavy step)
        # The orchestrator is sync; we can't hook mid-run here easily.
        report = orchestrator.run(document_text=document_text)

        # ── Done ─────────────────────────────────────────────────────────────
        job.status   = "done"
        job.stage    = "Complete"
        job.progress = 100
        job.result   = {
            "use_case_summary":              report.use_case_summary,
            "extracted_facts":               report.extracted_facts,
            "ai_definition_check":           report.ai_definition_check,
            "risk_classification":           report.risk_classification,
            "prohibited_practices_check":    report.prohibited_practices_check,
            "transparency_gpai_obligations": report.transparency_gpai_obligations,
            "roles":                         report.roles,
            "governance_observations":       report.governance_observations,
            "missing_information":           report.missing_information,
            "citations_by_source":           report.citations_by_source,
        }

    except Exception as exc:
        job.status = "error"
        job.stage  = "Failed"
        job.error  = str(exc)

    finally:
        # Clean up temp file and session
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        try:
            from retrieval.upload_ingest import clear_session  # noqa: PLC0415
            clear_session(f"web_{job_id[:8]}")
        except Exception:
            pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload a document and start the compliance analysis pipeline.

    Returns ``{"job_id": "..."}`` immediately.  Poll ``/api/status/{job_id}``
    until ``status == "done"`` or ``"error"``.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job_id        = str(uuid.uuid4())
    _jobs[job_id] = Job(job_id=job_id)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, job_id, tmp_path)

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll for job status.  Result is included once ``status == "done"``."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job


@app.get("/api/health")
async def health():
    return {"status": "ok", "jobs": len(_jobs)}
