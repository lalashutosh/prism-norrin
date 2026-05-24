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
    LLM_MAX_OUTPUT_TOKENS 16384
    EMBED_BASE_URL        https://containers.datacrunch.io/bge-m3/v1
    EMBED_MODEL           BAAI/bge-m3
    EMBED_API_KEY         (your DataCrunch token)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Prism API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Disk-backed job store ─────────────────────────────────────────────────────

_JOB_DIR = Path("logs/jobs")


class Job(BaseModel):
    job_id:   str
    status:   str = "pending"   # pending | extracting | analysing | waiting_llm | done | error
    stage:    str = "Queued"
    progress: int = 0           # 0–100
    result:   Optional[dict] = None
    error:    Optional[str]  = None


_jobs: dict[str, Job] = {}
_executor = ThreadPoolExecutor(max_workers=2)


def _persist_job(job: Job) -> None:
    """Write job state to logs/jobs/<job_id>.json so it survives server restarts."""
    try:
        _JOB_DIR.mkdir(parents=True, exist_ok=True)
        (_JOB_DIR / f"{job.job_id}.json").write_text(job.model_dump_json())
    except Exception:
        pass   # persistence is best-effort; never block the pipeline


def _load_persisted_jobs() -> None:
    """Load saved jobs on startup.

    Jobs that were in-progress when the server last died are marked as error
    (they cannot be auto-resumed without checkpoint serialisation).  Completed
    and error jobs are restored as-is so the UI can still view their results.
    """
    _JOB_DIR.mkdir(parents=True, exist_ok=True)
    for path in _JOB_DIR.glob("*.json"):
        try:
            job = Job.model_validate_json(path.read_text())
            if job.status in ("pending", "extracting", "analysing", "waiting_llm"):
                job.status = "error"
                job.stage  = "Interrupted"
                job.error  = (
                    "The server was restarted while this job was running. "
                    "Please re-submit to start a new analysis."
                )
            _jobs[job.job_id] = job
        except Exception:
            pass   # skip corrupt files


# Load on import so jobs are available before the first request arrives.
_load_persisted_jobs()


# ── Config helpers ────────────────────────────────────────────────────────────

def _cfg(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ── Pipeline runner (runs in thread pool) ─────────────────────────────────────

def _run_pipeline(job_id: str, tmp_path: str) -> None:
    """Execute the full Prism pipeline and update the job record."""
    from functools import partial

    import openai as _openai

    from agents.extraction_agent import run_extraction_agent
    from core.llm_client import make_llm_client, is_llm_available
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
        _persist_job(job)

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
        _persist_job(job)

        session_id = f"web_{job_id[:8]}"
        ingest_files(
            file_paths=[Path(tmp_path)],
            session_id=session_id,
            embed_fn=real_embed,
        )

        # ── Stage 3: extract text ────────────────────────────────────────────
        job.stage    = "Extracting document text…"
        job.progress = 40
        _persist_job(job)

        segments      = extract_file_text(Path(tmp_path))
        document_text = "\n\n".join(s["text"] for s in segments if s.get("text"))

        # ── Stage 4: orchestrator ────────────────────────────────────────────
        job.status   = "analysing"
        job.stage    = "Running compliance analysis…"
        job.progress = 55
        _persist_job(job)

        orchestrator = Orchestrator(
            retrieve_fn   = partial(retrieve, session_id=session_id),
            extraction_fn = partial(run_extraction_agent, llm_client=llm_client),
            llm_client    = llm_client,
            session_id    = session_id,
        )

        # Periodically update the stage to reflect LLM availability.
        # The LLM client's retry loop will block here when the LLM is down;
        # the status update lets the UI surface "waiting for LLM" correctly.
        import threading as _threading

        _stop_monitor = _threading.Event()

        def _monitor_llm():
            while not _stop_monitor.wait(timeout=3):
                if not is_llm_available():
                    job.status = "waiting_llm"
                    job.stage  = "LLM unavailable — waiting for reconnect…"
                    _persist_job(job)
                elif job.status == "waiting_llm":
                    # LLM came back
                    job.status = "analysing"
                    job.stage  = "Running compliance analysis…"
                    _persist_job(job)

        monitor_thread = _threading.Thread(target=_monitor_llm, daemon=True)
        monitor_thread.start()

        try:
            report = orchestrator.run(document_text=document_text)
        finally:
            _stop_monitor.set()

        # ── Done ─────────────────────────────────────────────────────────────
        job.status   = "done"
        job.stage    = "Complete"
        job.progress = 100
        job.result   = {
            # Sections 1–9 (narrative + scalar)
            "use_case_summary":              report.use_case_summary,
            "extracted_facts":               report.extracted_facts,
            "ai_definition_check":           report.ai_definition_check,
            "risk_classification":           report.risk_classification,
            "prohibited_practices_check":    report.prohibited_practices_check,
            "transparency_gpai_obligations": report.transparency_gpai_obligations,
            "roles":                         report.roles,
            "governance_observations":       report.governance_observations,
            "missing_information":           report.missing_information,
            # Section 10: per-dimension confidence with narrative
            "confidence_score":              report.confidence_score,
            # Section 11: claims grouped by epistemological label (programmatic)
            "evidence_separation":           report.evidence_separation,
            # Section 12: agent trace timeline (programmatic)
            "agent_trace":                   report.agent_trace,
            # Internal: citation registry
            "citations_by_source":           report.citations_by_source,
        }
        _persist_job(job)

    except Exception as exc:
        job.status = "error"
        job.stage  = "Failed"
        job.error  = str(exc)
        _persist_job(job)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        try:
            from retrieval.upload_ingest import clear_session   # noqa: PLC0415
            clear_session(f"web_{job_id[:8]}")
        except Exception:
            pass


# ── Shared job-launch helper ──────────────────────────────────────────────────

def _launch_job(content: bytes, suffix: str) -> str:
    """Write *content* to a temp file and enqueue a pipeline run. Returns job_id."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    job_id        = str(uuid.uuid4())
    _jobs[job_id] = Job(job_id=job_id)
    _persist_job(_jobs[job_id])

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, job_id, tmp_path)

    return job_id


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload a document and start the compliance analysis pipeline.

    Accepts PDF, DOCX, TXT, or MD files.
    Returns ``{"job_id": "..."}`` immediately.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix  = Path(file.filename).suffix or ".pdf"
    content = await file.read()
    job_id  = _launch_job(content, suffix)
    return {"job_id": job_id}


class TextAnalysisRequest(BaseModel):
    text:  str
    title: str = "Use case description"


@app.post("/api/analyze/text")
async def analyze_text(body: TextAnalysisRequest):
    """Submit plain text (instead of a file) for compliance analysis.

    The text is written to a temporary .txt file and runs the same pipeline
    as the file-upload path.  Returns ``{"job_id": "..."}`` immediately.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text body is empty.")

    content = body.text.encode("utf-8")
    job_id  = _launch_job(content, ".txt")
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll for job status.  Result is included once ``status == 'done'``."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job


@app.get("/api/jobs")
async def list_jobs():
    """Return all known job IDs and their statuses (newest-first approximation)."""
    return [
        {"job_id": j.job_id, "status": j.status, "stage": j.stage}
        for j in _jobs.values()
    ]


@app.get("/api/llm/status")
async def llm_status():
    """Return whether the LLM backend is currently reachable."""
    from core.llm_client import is_llm_available  # noqa: PLC0415
    return {
        "available": is_llm_available(),
        "message": "LLM reachable" if is_llm_available() else "LLM unreachable — pipeline is waiting",
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "jobs": len(_jobs)}
