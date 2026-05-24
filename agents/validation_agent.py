"""
agents/validation_agent.py
───────────────────────────
Validation agent: independently re-assesses weak claims from the analysis
phase and updates memory with confirmed, overturned, and unresolved findings.

══════════════════════════════════════════════════════════════════════════════
INTELLIGENCE — pure functions, no I/O, fully testable without API calls
══════════════════════════════════════════════════════════════════════════════
  identify_weak_claims(analysis_sections) -> list[WeakClaim]
  check_weak_claim_criteria(claim, dimension_id) -> (bool, str)
  build_claim_validation_prompt(weak_claim, facts, chunks) -> str
  parse_claim_validation_response(response_text, weak_claim)
      -> (ClaimStatus, str, list[str], Confidence, Label)
  build_overturned_claim(weak_claim, ...) -> OverturnedClaim
  _extract_json(text) -> dict

══════════════════════════════════════════════════════════════════════════════
ORCHESTRATION — coordinates calls, manages state, emits signals
══════════════════════════════════════════════════════════════════════════════
  _call_llm(prompt, system, client) -> str
  _validate_claim_worker(wc, facts, chunks, aug_ctx, llm_client)
      -> (claim_id, ClaimStatus, str, list[str], Confidence, Label)
  _get_analysis_sections(memory) -> dict[str, DimensionFinding]
  _get_chunks_for_claim(weak_claim, all_chunks) -> list[Chunk]
  run_validation_agent(memory, chunks, context, llm_client) -> Signal

PARALLEL EXECUTION
══════════════════
When the augmented context is available (the normal production path), EVERY
weak claim has pre-fetched authoritative chunks — so all claims can be
validated in parallel with a ThreadPoolExecutor.  The sequential retry path
(RetrievalSignal) is only needed on the legacy path when the general chunk
pool genuinely lacks authoritative coverage for a claim.

SQLite thread safety: the _write_lock in LogStore serialises concurrent writes
from worker threads (same as analysis_agent.py parallel path).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Union

from core.types import (
    AugmentedContext,
    Chunk,
    Claim,
    ClaimStatus,
    CompletionSignal,
    Confidence,
    DimensionFinding,
    FactSection,
    Label,
    OverturnedClaim,
    RetrievalSignal,
    ValidationFlag,
    ValidationSection,
    WeakClaim,
)
from core.augmentation import build_dimension_chunks
from core.logger import log_reasoning
from core.memory import ValidationAgentMemoryView
from prompts.validation_prompts import (
    CLAIM_VALIDATION_TEMPLATE,
    VALIDATION_SYSTEM_PROMPT,
)
from agents.analysis_agent import (
    _parse_confidence,
    _parse_label,
    _serialise_facts,
    _serialise_chunks,
    DIMENSION_KEYWORDS,
)

_logger = logging.getLogger(__name__)

# Maximum concurrent LLM calls for weak-claim validation.
# Each call is a short prompt (~1500 tokens) → short response (~300 tokens).
# 8 workers keeps GPU utilisation high without overwhelming a single-card server.
_VALIDATION_WORKERS = 8

# Critical dimensions where an ASSUMPTION label is always considered weak.
CRITICAL_DIMENSIONS = frozenset({
    "definition_check",
    "risk_classification",
    "prohibited_practices",
})


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def check_weak_claim_criteria(
    claim: Claim,
    dimension_id: str,
) -> tuple[bool, str]:
    """Determine whether a single claim should be flagged as weak.

    A claim is weak when any of the following conditions hold:
      LOW_CONFIDENCE  – claim.confidence is LOW or INSUFFICIENT
      ASSUMPTION      – claim.label is ASSUMPTION (always weak in critical dimensions;
                        weak in any dimension when confidence < HIGH)
      UNSUPPORTED     – claim.chunk_ids is empty (no citations)

    Returns (True, reason_code) or (False, "").
    """
    if claim.confidence in (Confidence.LOW, Confidence.INSUFFICIENT):
        return True, "LOW_CONFIDENCE"

    if claim.label == Label.ASSUMPTION:
        # ASSUMPTION on a critical dimension is always weak.
        if dimension_id in CRITICAL_DIMENSIONS:
            return True, "ASSUMPTION"
        # ASSUMPTION elsewhere is weak unless confidence is HIGH.
        if claim.confidence != Confidence.HIGH:
            return True, "ASSUMPTION"

    if not claim.chunk_ids:
        return True, "UNSUPPORTED"

    return False, ""


def identify_weak_claims(
    analysis_sections: dict[str, Optional[DimensionFinding]],
) -> list[WeakClaim]:
    """Walk all analysis sections and collect claims that need re-assessment.

    Parameters
    ----------
    analysis_sections : dict mapping dimension_id → DimensionFinding | None

    Returns
    -------
    list[WeakClaim] — deduplicated by claim_id, preserving encounter order.
    """
    seen_ids: set[str] = set()
    weak: list[WeakClaim] = []

    for dimension_id, section in analysis_sections.items():
        if section is None:
            continue
        for claim in section.claims:
            is_weak, reason = check_weak_claim_criteria(claim, dimension_id)
            if is_weak and claim.claim_id not in seen_ids:
                seen_ids.add(claim.claim_id)
                weak.append(
                    WeakClaim(
                        claim_id=claim.claim_id,
                        dimension_id=dimension_id,
                        claim_text=claim.text,
                        reason=reason,
                        original_confidence=claim.confidence,
                        original_label=claim.label,
                    )
                )
    return weak


def build_claim_validation_prompt(
    weak_claim: WeakClaim,
    facts: FactSection,
    chunks: list[Chunk],
) -> str:
    """Render the validation prompt for a single weak claim.

    Pure function — no I/O.
    """
    facts_text   = _serialise_facts(facts)
    chunks_text  = _serialise_chunks(chunks)
    return CLAIM_VALIDATION_TEMPLATE.format(
        dimension_id=weak_claim.dimension_id,
        claim_id=weak_claim.claim_id,
        claim_text=weak_claim.claim_text,
        weakness_reason=weak_claim.reason,
        original_confidence=weak_claim.original_confidence.value,
        original_label=weak_claim.original_label.value,
        chunks_text=chunks_text,
        facts_text=facts_text,
    )


def parse_claim_validation_response(
    response_text: str,
    weak_claim: WeakClaim,
) -> tuple[ClaimStatus, str, list[str], Confidence, Label]:
    """Parse the LLM verdict for a single claim.

    Returns
    -------
    (status, finding_text, supporting_chunk_ids, new_confidence, new_label)

    Falls back to (UNRESOLVED, ...) on parse failure so that the validation
    agent never crashes on bad LLM output.
    """
    raw = _extract_json(response_text)
    if not raw:
        return (
            ClaimStatus.UNRESOLVED,
            "Could not parse LLM response.",
            [],
            Confidence.INSUFFICIENT,
            Label.UNCERTAIN,
        )

    status_str = str(raw.get("status", "UNRESOLVED")).upper()
    try:
        status = ClaimStatus(status_str)
    except ValueError:
        status = ClaimStatus.UNRESOLVED

    finding      = str(raw.get("finding", ""))
    chunk_ids    = [str(c) for c in raw.get("supporting_chunk_ids", []) if c]
    new_conf     = _parse_confidence(raw.get("new_confidence", "INSUFFICIENT"))
    new_label    = _parse_label(raw.get("new_label", "UNCERTAIN"))

    return status, finding, chunk_ids, new_conf, new_label


def build_overturned_claim(
    weak_claim: WeakClaim,
    new_finding: str,
    new_confidence: Confidence,
    new_label: Label,
    new_chunk_ids: list[str],
) -> OverturnedClaim:
    """Construct an OverturnedClaim from a validation result."""
    return OverturnedClaim(
        claim_id=weak_claim.claim_id,
        dimension_id=weak_claim.dimension_id,
        original_claim_text=weak_claim.claim_text,
        new_finding=new_finding,
        new_confidence=new_confidence,
        new_label=new_label,
        new_chunk_ids=new_chunk_ids,
        status=ClaimStatus.OVERTURNED,
    )


# ── Private intelligence helpers ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Same three-strategy JSON extractor used by the analysis agent."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    block_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if block_match:
        try:
            result = json.loads(block_match.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, system: str, client: Any) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _validate_claim_worker(
    wc: WeakClaim,
    facts: FactSection,
    chunks: list[Chunk],
    aug_ctx: Optional[AugmentedContext],
    llm_client: Any,
) -> tuple[str, ClaimStatus, str, list[str], Confidence, Label]:
    """Pure thread-safe worker: build prompt → call LLM → parse response.

    Returns (claim_id, status, finding, new_chunk_ids, new_conf, new_label).
    Called from ThreadPoolExecutor workers; never touches mutable shared state.
    The log_reasoning decorator fires inside the worker thread — ContextVars
    are inherited from the parent thread in Python 3.7+ so the session_id and
    logger context are available, and LogStore._write_lock prevents data races.
    """
    relevant = _get_chunks_for_claim(wc, chunks, aug_ctx=aug_ctx)
    prompt   = build_claim_validation_prompt(wc, facts, relevant)
    _logged_llm = log_reasoning(agent="validation", dimension=wc.dimension_id)(_call_llm)
    response_text = _logged_llm(prompt, VALIDATION_SYSTEM_PROMPT, llm_client)
    status, finding, new_chunk_ids, new_conf, new_label = (
        parse_claim_validation_response(response_text, wc)
    )
    return wc.claim_id, status, finding, new_chunk_ids, new_conf, new_label


def _get_analysis_sections(
    memory: ValidationAgentMemoryView,
) -> dict[str, Optional[DimensionFinding]]:
    """Collect all six analysis sections from the memory proxy."""
    return {
        "definition_check":    memory.definition_check,
        "risk_classification": memory.risk_classification,
        "prohibited_practices": memory.prohibited_practices,
        "transparency":        memory.transparency,
        "roles":               memory.roles,
        "governance":          memory.governance,
    }


def _get_chunks_for_claim(
    weak_claim: WeakClaim,
    all_chunks: list[Chunk],
    aug_ctx: Optional[AugmentedContext] = None,
) -> list[Chunk]:
    """Return chunks relevant to *weak_claim* for validation.

    Priority order:
      1. **Augmented context** (when available): uses the pre-computed,
         dimension-specific legal chunks from the augmentation layer.
         These are already filtered by source_type so Article 5 official
         guidance cannot bleed into non-prohibited-practices claims.
         User-document chunks are appended (up to 4) so the validator
         can also assess FACT labels.
      2. **Legacy path** (fallback): keyword-filters the general chunk pool
         by dimension keywords.  Used when augmented context is absent
         (no session_id, tests, or augmentation error) or returns nothing.

    Falls back to all_chunks if neither path yields results.
    """
    if aug_ctx is not None:
        # Augmented path: dimension-specific legal chunks (pre-filtered source_type)
        aug_chunks = build_dimension_chunks(aug_ctx, weak_claim.dimension_id)
        if aug_chunks:
            # Append a handful of user-doc chunks so FACT claims can be validated.
            user_supplement = aug_ctx.user_chunks[:4]
            # Deduplicate: aug_chunks are all legal; user_chunks are all uploaded_doc.
            return aug_chunks + user_supplement

    # Legacy path: keyword filtering from general pool
    keywords = DIMENSION_KEYWORDS.get(weak_claim.dimension_id, [])
    relevant = [
        c for c in all_chunks
        if any(kw.lower() in c.text.lower() for kw in keywords)
    ]
    return relevant if relevant else all_chunks


def run_validation_agent(
    memory: ValidationAgentMemoryView,
    chunks: list[Chunk],
    context: dict,
    llm_client: Any = None,
) -> Union[RetrievalSignal, CompletionSignal]:
    """Drive the validation agent through all identified weak claims.

    ORCHESTRATION ENTRY POINT.

    On each invocation the agent:
      1. Re-identifies weak claims from current memory (idempotent).
      2. Skips claims already in context["processed_claim_ids"].
      3. For the next unprocessed claim: checks if relevant chunks exist.
         - If not AND retry_counts[claim_id] < 2: emits RetrievalSignal.
         - If not AND retry limit hit: marks UNRESOLVED and continues.
      4. Calls the LLM for the claim.
      5. Records the result.
      6. After all claims processed: writes to memory and returns CompletionSignal.

    context keys consumed:
      "processed_claim_ids" : set[str]  — claim IDs already handled
      "retry_counts"        : dict[str, int]  — retries per claim_id
      "max_retrievals_reached" : set[str]  — claim_ids at retry limit
    """
    if llm_client is None:
        import anthropic
        llm_client = anthropic.Anthropic()

    facts = memory.facts
    if facts is None:
        raise ValueError("FactSection not available in memory.")

    # Read the pre-computed augmented context once; passed through to
    # _get_chunks_for_claim so each claim gets dimension-specific legal
    # chunks rather than a keyword-filtered slice of the general pool.
    aug_ctx: Optional[AugmentedContext] = memory.augmented_context

    analysis_sections = _get_analysis_sections(memory)
    all_weak          = identify_weak_claims(analysis_sections)

    # setdefault() is used here rather than direct assignment so that calling
    # run_validation_agent again after a RetrievalSignal re-uses the same
    # mutable objects that were populated in the previous invocation.
    # The orchestrator never resets these between signal cycles — it is the
    # agent's responsibility to skip already-processed claims via processed_ids.
    processed_ids: set[str]   = context.setdefault("processed_claim_ids", set())
    retry_counts:  dict[str, int] = context.setdefault("retry_counts", {})
    max_reached:   set[str]   = context.setdefault("max_retrievals_reached", set())

    # Results are accumulated across invocations via context (survives signal cycles).
    # Writing only happens after the for-loop finishes all weak claims.
    overturned_claims: list[OverturnedClaim]  = context.setdefault("overturned_claims", [])
    flags:             list[ValidationFlag]   = context.setdefault("flags", [])
    unresolved_ids:    list[str]              = context.setdefault("unresolved_ids", [])

    # ── Classify unprocessed claims into two buckets ─────────────────────────
    # Parallel bucket  : claim has authoritative chunks right now → can validate
    # Retrieval bucket : no authoritative chunks → needs RetrievalSignal retry
    parallel_bucket:   list[WeakClaim] = []
    retrieval_bucket:  list[WeakClaim] = []

    for wc in all_weak:
        if wc.claim_id in processed_ids:
            continue
        relevant = _get_chunks_for_claim(wc, chunks, aug_ctx=aug_ctx)
        has_auth = any(
            c.source_type in ("legislation", "official_guidance") for c in relevant
        )
        if has_auth or wc.claim_id in max_reached:
            parallel_bucket.append(wc)
        else:
            retrieval_bucket.append(wc)

    # ── Handle retrieval-needed claims first (sequential, may return early) ──
    # This preserves the original RetrievalSignal contract: if a claim still
    # has no authoritative evidence and hasn't exhausted retries, we stop and
    # ask the orchestrator to fetch more chunks before continuing.
    for wc in retrieval_bucket:
        cid = wc.claim_id
        retries = retry_counts.get(cid, 0)
        if retries < 2:
            return RetrievalSignal(
                query=(
                    f"{wc.dimension_id} {wc.claim_text[:120]} EU AI Act evidence"
                ),
                filters={
                    "dimension":    wc.dimension_id,
                    "source_types": ["legislation", "official_guidance"],
                },
                dimension=cid,
            )
        # Retry limit exhausted — move to parallel bucket so it still gets assessed
        # with whatever chunks exist (may result in UNRESOLVED, which is correct).
        max_reached.add(cid)
        parallel_bucket.append(wc)

    # ── Validate all ready claims in parallel ─────────────────────────────────
    # Workers are pure: (prompt → LLM → parse → return tuple).
    # All state mutations happen in the main thread after the pool joins.
    # The ThreadPoolExecutor is only created when there are claims to process
    # so the overhead is zero when the agent is called on an already-clean state.
    if parallel_bucket:
        n_workers = min(len(parallel_bucket), _VALIDATION_WORKERS)
        raw_results: dict[str, tuple] = {}   # claim_id → (status, finding, ...)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_wc = {
                pool.submit(
                    _validate_claim_worker,
                    wc, facts, chunks, aug_ctx, llm_client,
                ): wc
                for wc in parallel_bucket
            }
            for future in as_completed(future_to_wc):
                wc = future_to_wc[future]
                try:
                    cid, status, finding, new_chunk_ids, new_conf, new_label = (
                        future.result()
                    )
                except Exception as exc:
                    _logger.warning(
                        "validation worker failed for %s: %s", wc.claim_id, exc
                    )
                    cid, status, finding, new_chunk_ids, new_conf, new_label = (
                        wc.claim_id,
                        ClaimStatus.UNRESOLVED,
                        f"Worker exception: {exc}",
                        [],
                        Confidence.INSUFFICIENT,
                        Label.UNCERTAIN,
                    )
                raw_results[cid] = (status, finding, new_chunk_ids, new_conf, new_label)

        # ── Record outcomes in deterministic order (all_weak preserves analysis order)
        for wc in parallel_bucket:
            cid = wc.claim_id
            status, finding, new_chunk_ids, new_conf, new_label = raw_results[cid]

            if status == ClaimStatus.OVERTURNED:
                overturned_claims.append(
                    build_overturned_claim(wc, finding, new_conf, new_label, new_chunk_ids)
                )
            elif status == ClaimStatus.UNRESOLVED:
                unresolved_ids.append(cid)

            flags.append(
                ValidationFlag(
                    claim_id=cid,
                    dimension_id=wc.dimension_id,
                    status=status,
                    notes=finding,
                    new_chunk_ids=new_chunk_ids,
                )
            )
            processed_ids.add(cid)

    # All weak claims processed — determine overall confidence and write.
    has_unresolved = bool(unresolved_ids)
    has_overturned = bool(overturned_claims)

    # Overall validation confidence reflects the worst outcome encountered:
    #   LOW    — any claim could not be resolved (still uncertain after retry)
    #   MEDIUM — at least one claim was overturned (evidence quality improved)
    #   HIGH   — every weak claim was confirmed (original analysis was sound)
    if has_unresolved:
        overall = Confidence.LOW
    elif has_overturned:
        overall = Confidence.MEDIUM
    else:
        overall = Confidence.HIGH

    validation_section = ValidationSection(
        flags=flags,
        overall_confidence=overall,
        summary=(
            f"{len(all_weak)} weak claim(s) reviewed. "
            f"{len(overturned_claims)} overturned, "
            f"{len(unresolved_ids)} unresolved."
        ),
    )

    memory.write_validation_flags(validation_section)
    memory.write_weak_claims(all_weak)
    memory.write_overturned_claims(overturned_claims)

    return CompletionSignal(agent="validation", message="Validation complete.")
