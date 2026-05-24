"""
agents/extraction_agent.py
──────────────────────────
Extraction agent: reads raw document text and produces a structured FactSection.

This is the pipeline's entry point for user-supplied content.  It runs BEFORE
any retrieval so that the orchestrator can build a focused initial query.

══════════════════════════════════════════════════════════════════════════════
INTELLIGENCE — pure functions, no I/O, fully testable without API calls
══════════════════════════════════════════════════════════════════════════════
  build_extraction_prompt(document_text)     → str
  parse_extraction_response(response_text)   → FactSection
  _extract_json(text)                        → dict

══════════════════════════════════════════════════════════════════════════════
ORCHESTRATION — coordinates LLM call, returns typed result
══════════════════════════════════════════════════════════════════════════════
  run_extraction_agent(document_text, llm_client) → FactSection

The Orchestrator wires this in as:
    from functools import partial
    extraction_fn = partial(run_extraction_agent, llm_client=my_client)
    orchestrator  = Orchestrator(extraction_fn=extraction_fn, ...)
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from core.logger import log_reasoning
from core.types import FactSection


# ── System prompt ─────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are a legal-AI compliance analyst specialising in the EU AI Act.

Your task is to extract structured facts about an AI system from a document.
You output ONLY a valid JSON object — no prose before or after the JSON block.

Extract exactly these fields (use null for unknown optional fields, [] for
unknown lists):

{
  "use_case_name":       "<short name for the AI system or use case>",
  "description":         "<2-4 sentence factual description of what the system does>",
  "industry":            "<industry sector, e.g. healthcare, finance, manufacturing>",
  "ai_capabilities":     ["<capability 1>", "<capability 2>", ...],
  "data_inputs":         ["<input type 1>", "<input type 2>", ...],
  "outputs":             ["<output type 1>", "<output type 2>", ...],
  "deployment_context":  "<where / how the system is deployed>",
  "affected_persons":    ["<group 1>", "<group 2>", ...],
  "existing_oversight":  "<any human oversight or safeguards mentioned, or null>",
  "vendor_or_developer": "<developer or vendor name, or null>"
}

Be precise and factual.  Do not infer or speculate beyond what the document states.
"""


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def build_extraction_prompt(document_text: str) -> str:
    """Render the user-facing extraction prompt.

    Truncates the document to 12 000 characters to stay within typical context
    windows when dealing with long PDFs.  The most relevant information is
    almost always in the first portion of a use-case document.

    Pure function — no I/O.
    """
    MAX_CHARS = 12_000  # ~3k tokens; well within 131k context window
    truncated = document_text[:MAX_CHARS]
    suffix = "\n\n[... document truncated ...]" if len(document_text) > MAX_CHARS else ""

    return (
        "Extract structured facts about the AI system described below.\n\n"
        "=== DOCUMENT START ===\n"
        f"{truncated}{suffix}\n"
        "=== DOCUMENT END ===\n\n"
        "Respond with ONLY the JSON object described in your instructions."
    )


def parse_extraction_response(response_text: str) -> FactSection:
    """Parse the LLM's JSON response into a FactSection.

    Falls back to a minimal FactSection if the response cannot be parsed,
    so the pipeline always proceeds rather than crashing on a bad LLM output.

    Pure function — no I/O.
    """
    raw = _extract_json(response_text)

    def _str(key: str, fallback: str = "") -> str:
        val = raw.get(key)
        return str(val).strip() if val and str(val).strip().lower() not in (
            "unknown", "none", "n/a", "null", ""
        ) else fallback

    def _opt(key: str) -> Optional[str]:
        val = _str(key)
        return val if val else None

    def _lst(key: str) -> list[str]:
        val = raw.get(key)
        if isinstance(val, list):
            return [str(v).strip() for v in val if v and str(v).strip()]
        if isinstance(val, str) and val.strip():
            # Comma-separated fallback
            return [v.strip() for v in val.split(",") if v.strip()]
        return []

    return FactSection(
        use_case_name       = _str("use_case_name", "Unidentified AI System"),
        description         = _str("description",   "No description extracted."),
        industry            = _opt("industry"),
        ai_capabilities     = _lst("ai_capabilities"),
        data_inputs         = _lst("data_inputs"),
        outputs             = _lst("outputs"),
        deployment_context  = _opt("deployment_context"),
        affected_persons    = _lst("affected_persons"),
        existing_oversight  = _opt("existing_oversight"),
        vendor_or_developer = _opt("vendor_or_developer"),
        additional_facts    = {},
        source_chunk_ids    = [],   # populated by the orchestrator from ingested chunks
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Three-strategy JSON extractor (same pattern as analysis/synthesis agents)."""
    # Strategy 1: the whole response is valid JSON.
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: JSON inside a ```json … ``` code fence.
    block = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if block:
        try:
            result = json.loads(block.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: first {...} block in the response.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            result = json.loads(brace.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

@log_reasoning(agent="extraction")
def _call_llm(prompt: str, system: str, client: Any) -> str:
    """Call the LLM and return the raw response text.

    Uses Anthropic's wire format — works with the native anthropic.Anthropic()
    client or with core.llm_client.OpenAIAdapter for Gemini / OpenAI.
    """
    response = client.messages.create(
        model     = "claude-sonnet-4-6",   # overridden by OpenAIAdapter.model
        max_tokens= 1024,
        system    = system,
        messages  = [{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def run_extraction_agent(
    document_text: str,
    llm_client:    Any = None,
) -> FactSection:
    """Extract structured facts from *document_text* and return a FactSection.

    This is the orchestrator's extraction_fn entry point.  Wire it via partial:

        from functools import partial
        from agents.extraction_agent import run_extraction_agent
        extraction_fn = partial(run_extraction_agent, llm_client=my_client)

    Parameters
    ──────────
    document_text  Raw text of the user's uploaded document (post-PDF extraction).
    llm_client     Anthropic client or OpenAIAdapter.  If None, creates a real
                   anthropic.Anthropic() client automatically.

    Returns
    ───────
    FactSection — fully typed, ready for memory.write_facts().
    """
    if llm_client is None:
        import anthropic  # noqa: PLC0415
        llm_client = anthropic.Anthropic()

    prompt        = build_extraction_prompt(document_text)
    response_text = _call_llm(prompt, _EXTRACTION_SYSTEM_PROMPT, llm_client)
    return parse_extraction_response(response_text)
