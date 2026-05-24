"""
agents/analysis_agent.py
─────────────────────────
Analysis agent: reasons over retrieved chunks and produces a structured
assessment across six EU AI Act legal dimensions.

══════════════════════════════════════════════════════════════════════════════
INTELLIGENCE — pure functions, no I/O, fully testable without API calls
══════════════════════════════════════════════════════════════════════════════
  check_evidence_sufficiency(chunks, dimension_id) -> (bool, str | None)
  formulate_retrieval_query(dimension_id, facts, existing_chunks) -> (str, dict)
  build_dimension_prompt(facts, chunks, dimension_id, refined_context) -> str
  parse_dimension_response(response_text, dimension_id) -> DimensionFinding
  score_overall_confidence(claims) -> Confidence
  _extract_json(text) -> dict
  _make_insufficient_finding(dimension_id) -> DimensionFinding subtype
  _make_claim(raw, dimension_id, idx) -> Claim

══════════════════════════════════════════════════════════════════════════════
ORCHESTRATION — coordinates calls, manages state, emits signals
══════════════════════════════════════════════════════════════════════════════
  _is_dimension_done(memory, dimension_id) -> bool
  _write_dimension(memory, dimension_id, finding) -> None
  _filter_chunks_for_dimension(chunks, dimension_id) -> list[Chunk]
  _call_llm(prompt, system, client) -> str
  run_analysis_agent(memory, chunks, context, llm_client) -> Signal
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Union

from core.types import (
    AugmentedContext,
    Chunk,
    Claim,
    ClaimStatus,
    Confidence,
    DefinitionSection,
    DimensionFinding,
    GovernanceSection,
    Label,
    ProhibitedSection,
    RetrievalSignal,
    CompletionSignal,
    RiskLevel,
    RiskSection,
    RolesSection,
    TransparencySection,
    FactSection,
)
from core.augmentation import (
    build_dimension_chunks,
    serialise_augmented_context,
)
from core.logger import log_reasoning
from core.memory import AnalysisAgentMemoryView
from prompts.analysis_prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    DIMENSION_KEYWORDS,
    DIMENSION_PROMPTS,
    DIMENSION_RETRIEVAL_TEMPLATES,
)

# Processing order is fixed; defines sequential dimension assessment.
DIMENSION_ORDER: list[str] = [
    "definition_check",
    "risk_classification",
    "prohibited_practices",
    "transparency",
    "roles",
    "governance",
]

# Minimum legislation/official_guidance chunks needed per dimension.
MIN_AUTHORITATIVE_CHUNKS: dict[str, int] = {
    "definition_check":    1,
    "risk_classification": 1,
    "prohibited_practices": 1,
    "transparency":        1,
    "roles":               1,
    "governance":          1,
}

AUTHORITATIVE_SOURCE_TYPES = frozenset({"legislation", "official_guidance"})


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def check_evidence_sufficiency(
    chunks: list[Chunk],
    dimension_id: str,
) -> tuple[bool, Optional[str]]:
    """Return (True, None) when chunks are sufficient to assess *dimension_id*.

    Sufficiency requires:
      1. At least one chunk contains a keyword relevant to the dimension.
      2. At least MIN_AUTHORITATIVE_CHUNKS[dimension_id] chunks come from
         an authoritative source (legislation / official_guidance).

    Returns (False, reason_string) when either condition fails.
    """
    if not chunks:
        return False, f"No chunks available for dimension '{dimension_id}'"

    keywords = DIMENSION_KEYWORDS.get(dimension_id, [])
    relevant = [
        c for c in chunks
        if any(kw.lower() in c.text.lower() for kw in keywords)
    ]
    if not relevant:
        return False, (
            f"No chunks contain keywords for '{dimension_id}': "
            f"{keywords[:4]!r}…"
        )

    authoritative = [c for c in relevant if c.source_type in AUTHORITATIVE_SOURCE_TYPES]
    min_needed = MIN_AUTHORITATIVE_CHUNKS.get(dimension_id, 1)
    if len(authoritative) < min_needed:
        return False, (
            f"Need ≥{min_needed} authoritative chunk(s) for '{dimension_id}', "
            f"found {len(authoritative)}"
        )

    return True, None


def formulate_retrieval_query(
    dimension_id: str,
    facts: FactSection,
    existing_chunks: list[Chunk],
) -> tuple[str, dict]:
    """Build a targeted retrieval query for *dimension_id*.

    The query combines a dimension-specific template with key facts about
    the use case so the retrieval layer returns focused results.
    Returns (query_string, filters_dict).
    """
    context_hint = " ".join(
        filter(None, [
            facts.use_case_name,
            facts.industry,
            facts.deployment_context,
        ])
    )[:200]  # keep query concise

    template = DIMENSION_RETRIEVAL_TEMPLATES.get(
        dimension_id,
        f"EU AI Act {dimension_id} {{context}}",
    )
    query = template.format(context=context_hint).strip()

    filters: dict = {
        "source_types": ["legislation", "official_guidance"],
        "dimension":    dimension_id,
    }
    return query, filters


def build_dimension_prompt(
    facts: FactSection,
    chunks: list[Chunk],
    dimension_id: str,
    refined_context: str = "",
    augmented_context: Optional[AugmentedContext] = None,
) -> str:
    """Render the full user-turn prompt for a single dimension assessment.

    When *augmented_context* is provided the RETRIEVED CHUNKS block is
    replaced with the pre-mapped context produced by the augmentation layer.
    This renders both the legal provisions and the user-document facts that
    triggered each mapping, and explicitly lists the only valid chunk_ids.

    Pure function — takes structured data, returns a string.  No I/O or
    side effects.
    """
    facts_text = _serialise_facts(facts)

    if augmented_context is not None:
        aug_text = serialise_augmented_context(augmented_context, dimension_id)
        chunks_text = aug_text if aug_text else _serialise_chunks(chunks)
    else:
        chunks_text = _serialise_chunks(chunks)

    context_block = (
        f"ADDITIONAL CONTEXT FROM LOOP REFINEMENT\n{refined_context}"
        if refined_context.strip()
        else ""
    )
    template = DIMENSION_PROMPTS.get(dimension_id, "")
    if not template:
        raise ValueError(f"No prompt template found for dimension '{dimension_id}'")
    return template.format(
        facts_text=facts_text,
        chunks_text=chunks_text,
        refined_context=context_block,
    )


def parse_dimension_response(
    response_text: str,
    dimension_id: str,
) -> DimensionFinding:
    """Parse an LLM response string into the correct DimensionFinding subtype.

    Falls back to an INSUFFICIENT finding when the response cannot be parsed
    or is missing required fields.  Never raises — always returns a valid type.
    """
    raw = _extract_json(response_text)
    if not raw:
        return _make_insufficient_finding(dimension_id)

    try:
        raw_claims = raw.get("claims", [])
        claims = [_make_claim(rc, dimension_id, i) for i, rc in enumerate(raw_claims)]
        confidence = _parse_confidence(raw.get("confidence", "INSUFFICIENT"))
        summary = str(raw.get("summary", ""))

        if dimension_id == "definition_check":
            return DefinitionSection(
                dimension_id=dimension_id,
                claims=claims,
                confidence=confidence,
                summary=summary,
                is_ai_system=raw.get("is_ai_system"),
            )
        if dimension_id == "risk_classification":
            raw_level = raw.get("risk_level", "unknown")
            risk_level = _parse_risk_level(raw_level)
            return RiskSection(
                dimension_id=dimension_id,
                claims=claims,
                confidence=confidence,
                summary=summary,
                risk_level=risk_level,
            )
        if dimension_id == "prohibited_practices":
            return ProhibitedSection(
                dimension_id=dimension_id,
                claims=claims,
                confidence=confidence,
                summary=summary,
                triggered_articles=raw.get("triggered_articles", []),
                prohibited=raw.get("prohibited"),
            )
        if dimension_id == "transparency":
            return TransparencySection(
                dimension_id=dimension_id,
                claims=claims,
                confidence=confidence,
                summary=summary,
                applies_to_gpai=bool(raw.get("applies_to_gpai", False)),
                labelling_required=bool(raw.get("labelling_required", False)),
                notification_required=bool(raw.get("notification_required", False)),
            )
        if dimension_id == "roles":
            return RolesSection(
                dimension_id=dimension_id,
                claims=claims,
                confidence=confidence,
                summary=summary,
                is_provider=bool(raw.get("is_provider", False)),
                is_deployer=bool(raw.get("is_deployer", False)),
                is_both=bool(raw.get("is_both", False)),
            )
        if dimension_id == "governance":
            return GovernanceSection(
                dimension_id=dimension_id,
                claims=claims,
                confidence=confidence,
                summary=summary,
                documentation_required=bool(raw.get("documentation_required", False)),
                oversight_required=bool(raw.get("oversight_required", False)),
                monitoring_required=bool(raw.get("monitoring_required", False)),
            )
    except Exception:  # noqa: BLE001
        pass  # fall through to INSUFFICIENT

    return _make_insufficient_finding(dimension_id)


def score_overall_confidence(claims: list[Claim]) -> Confidence:
    """Derive an overall confidence level from a list of claims.

    Rules (in priority order):
      1. Any INSUFFICIENT claim → INSUFFICIENT overall.
      2. ≥80 % HIGH → HIGH.
      3. ≥50 % HIGH or MEDIUM → MEDIUM.
      4. Otherwise → LOW.
    """
    if not claims:
        return Confidence.INSUFFICIENT

    counts: dict[Confidence, int] = {c: 0 for c in Confidence}
    for claim in claims:
        counts[claim.confidence] = counts.get(claim.confidence, 0) + 1

    # Rule 1 (highest priority): any INSUFFICIENT claim poisons the whole section.
    if counts[Confidence.INSUFFICIENT] > 0:
        return Confidence.INSUFFICIENT

    total = len(claims)
    high_ratio = counts[Confidence.HIGH] / total
    high_med_ratio = (counts[Confidence.HIGH] + counts[Confidence.MEDIUM]) / total

    # Rule 2: ≥80% HIGH → the section is highly reliable.
    if high_ratio >= 0.8:
        return Confidence.HIGH
    # Rule 3: ≥50% HIGH or MEDIUM → the section is moderately reliable.
    if high_med_ratio >= 0.5:
        return Confidence.MEDIUM
    # Rule 4: less than half the claims have useful confidence → LOW.
    return Confidence.LOW


# ── Private intelligence helpers ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from *text*.

    Tries three strategies:
      1. Direct json.loads on the whole string.
      2. Extract from a ```json ... ``` code block.
      3. Regex for the first { ... } span (greedy).
    Returns an empty dict on total failure.
    """
    # Strategy 1 – direct parse: LLM followed the "JSON only" instruction.
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2 – markdown code block: LLM wrapped JSON in a ```json fence.
    block_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if block_match:
        try:
            result = json.loads(block_match.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3 – greedy brace match: JSON is embedded inside surrounding prose.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # All three strategies failed — return empty dict so callers fall back to INSUFFICIENT.
    return {}


def _make_insufficient_finding(dimension_id: str) -> DimensionFinding:
    """Return the correct subtype for *dimension_id* with INSUFFICIENT confidence."""
    base = dict(
        dimension_id=dimension_id,
        claims=[],
        confidence=Confidence.INSUFFICIENT,
        summary=f"Insufficient evidence to assess dimension '{dimension_id}'.",
    )
    if dimension_id == "definition_check":
        return DefinitionSection(**base, is_ai_system=None)
    if dimension_id == "risk_classification":
        return RiskSection(**base, risk_level=RiskLevel.UNKNOWN)
    if dimension_id == "prohibited_practices":
        return ProhibitedSection(**base, triggered_articles=[], prohibited=None)
    if dimension_id == "transparency":
        return TransparencySection(**base)
    if dimension_id == "roles":
        return RolesSection(**base)
    if dimension_id == "governance":
        return GovernanceSection(**base)
    # Fallback for unknown dimension_ids (shouldn't happen in production)
    return DimensionFinding(**base)


def _make_claim(raw: Any, dimension_id: str, idx: int) -> Claim:
    """Construct a Claim from a raw dict parsed from an LLM response.

    Applies safe defaults for every field so malformed LLM output does not
    crash the parser.
    """
    if not isinstance(raw, dict):
        raw = {}
    return Claim(
        claim_id=str(raw.get("claim_id", f"{dimension_id}_{idx}")),
        text=str(raw.get("text", "")),
        label=_parse_label(raw.get("label", "UNCERTAIN")),
        confidence=_parse_confidence(raw.get("confidence", "LOW")),
        chunk_ids=[str(c) for c in raw.get("chunk_ids", []) if c],
        is_weak=bool(raw.get("is_weak", False)),
        weak_reason=raw.get("weak_reason") or None,
    )


def _parse_label(value: Any) -> Label:
    try:
        return Label(str(value).upper())
    except ValueError:
        return Label.UNCERTAIN


def _parse_confidence(value: Any) -> Confidence:
    try:
        return Confidence(str(value).upper())
    except ValueError:
        return Confidence.INSUFFICIENT


def _parse_risk_level(value: Any) -> RiskLevel:
    try:
        return RiskLevel(str(value).lower())
    except ValueError:
        return RiskLevel.UNKNOWN


def _serialise_facts(facts: FactSection) -> str:
    """Convert a FactSection to a readable text block for prompt injection."""
    lines = [
        f"Use case: {facts.use_case_name}",
        f"Description: {facts.description}",
    ]
    if facts.industry:
        lines.append(f"Industry: {facts.industry}")
    if facts.ai_capabilities:
        lines.append(f"AI capabilities: {', '.join(facts.ai_capabilities)}")
    if facts.data_inputs:
        lines.append(f"Data inputs: {', '.join(facts.data_inputs)}")
    if facts.outputs:
        lines.append(f"Outputs: {', '.join(facts.outputs)}")
    if facts.deployment_context:
        lines.append(f"Deployment context: {facts.deployment_context}")
    if facts.affected_persons:
        lines.append(f"Affected persons: {', '.join(facts.affected_persons)}")
    if facts.existing_oversight:
        lines.append(f"Existing oversight: {facts.existing_oversight}")
    if facts.vendor_or_developer:
        lines.append(f"Vendor/developer: {facts.vendor_or_developer}")
    return "\n".join(lines)


def _serialise_chunks(chunks: list[Chunk]) -> str:
    """Render a list of Chunks as a numbered block for prompt injection."""
    if not chunks:
        return "(no chunks available)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        article = f" [{chunk.article_id}]" if chunk.article_id else ""
        parts.append(
            f"[{i}] chunk_id={chunk.chunk_id} source={chunk.source_type}{article}\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_dimension_done(memory: AnalysisAgentMemoryView, dimension_id: str) -> bool:
    """Return True if the analysis section for *dimension_id* has been written."""
    section = {
        "definition_check":    memory.definition_check,
        "risk_classification": memory.risk_classification,
        "prohibited_practices": memory.prohibited_practices,
        "transparency":        memory.transparency,
        "roles":               memory.roles,
        "governance":          memory.governance,
    }.get(dimension_id)
    return section is not None


def _write_dimension(
    memory: AnalysisAgentMemoryView,
    dimension_id: str,
    finding: DimensionFinding,
) -> None:
    """Dispatch to the correct proxy write method for *dimension_id*."""
    dispatch = {
        "definition_check":    memory.write_definition,
        "risk_classification": memory.write_risk,
        "prohibited_practices": memory.write_prohibited,
        "transparency":        memory.write_transparency,
        "roles":               memory.write_roles,
        "governance":          memory.write_governance,
    }
    writer = dispatch.get(dimension_id)
    if writer is None:
        raise ValueError(f"Unknown dimension_id '{dimension_id}'")
    writer(finding)


def _filter_chunks_for_dimension(
    chunks: list[Chunk],
    dimension_id: str,
) -> list[Chunk]:
    """Return chunks relevant to *dimension_id* via keyword matching.

    Falls back to returning all chunks if no keyword match is found, so
    the agent is never starved of context on its first invocation.
    """
    keywords = DIMENSION_KEYWORDS.get(dimension_id, [])
    relevant = [
        c for c in chunks
        if any(kw.lower() in c.text.lower() for kw in keywords)
    ]
    return relevant if relevant else chunks


def _call_llm(prompt: str, system: str, client: Any) -> str:
    """Invoke the LLM and return the first text content block.

    Kept thin so that tests can substitute a mock client with a
    `.messages.create(...)` interface.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def run_analysis_agent(
    memory: AnalysisAgentMemoryView,
    chunks: list[Chunk],
    context: dict,
    llm_client: Any = None,
) -> Union[RetrievalSignal, CompletionSignal]:
    """Drive the analysis agent through all six dimensions.

    ORCHESTRATION ENTRY POINT — contains no prompts, no legal reasoning,
    no confidence scoring.  All intelligence is delegated to the pure
    functions above.

    Parameters
    ----------
    memory : AnalysisAgentMemoryView
        Proxy for the current SessionMemory.  Created fresh by the
        orchestrator on every invocation (including re-invocations after
        a RetrievalSignal).
    chunks : list[Chunk]
        All chunks retrieved so far in this session.  The list grows
        across signal cycles as the orchestrator adds new results.
    context : dict
        Orchestrator-managed per-cycle metadata:
          "max_retrievals_reached" : set[str]
              Dimension IDs for which the retry limit (2) has been hit.
              The agent must not emit a RetrievalSignal for these; instead
              it writes an INSUFFICIENT finding.
          "refined_context" : str
              Additional guidance injected by the synthesis LoopSignal.
    llm_client : optional
        Anthropic client instance.  None → create a real client.
        Tests pass a mock here to avoid API calls.

    Returns
    -------
    RetrievalSignal  when evidence for a dimension is insufficient and
                     the retry limit has not been reached.
    CompletionSignal when all six sections have been written to memory.
    """
    if llm_client is None:
        import anthropic  # deferred so tests never need the package
        llm_client = anthropic.Anthropic()

    facts = memory.facts
    if facts is None:
        raise ValueError(
            "FactSection not available in memory; extraction must run first."
        )

    max_reached: set[str]       = context.get("max_retrievals_reached", set())
    refined_context: str        = context.get("refined_context", "")
    aug_ctx: Optional[AugmentedContext] = memory.augmented_context

    for dimension_id in DIMENSION_ORDER:
        # Resume detection — skip already-written sections.
        if _is_dimension_done(memory, dimension_id):
            continue

        if aug_ctx is not None:
            # ── Augmented path ───────────────────────────────────────────────
            # The augmentation layer pre-fetched the right legal chunks for
            # this dimension with the correct source_type filter (no Article 5
            # dilution).  Use these as the primary evidence source.
            # User-doc chunks are embedded in the serialised augmented context
            # so agents can cite them directly as FACT claims.
            aug_chunks = build_dimension_chunks(aug_ctx, dimension_id)
            # Check sufficiency; if the augmented context has authoritative
            # chunks we proceed even without a RetrievalSignal cycle.
            if aug_chunks:
                prompt = build_dimension_prompt(
                    facts, aug_chunks, dimension_id, refined_context,
                    augmented_context=aug_ctx,
                )
                _logged_llm = log_reasoning(agent="analysis", dimension=dimension_id)(_call_llm)
                response_text = _logged_llm(prompt, ANALYSIS_SYSTEM_PROMPT, llm_client)
                finding = parse_dimension_response(response_text, dimension_id)
                _write_dimension(memory, dimension_id, finding)
                continue

            # Augmentation returned nothing for this dimension — fall through
            # to the legacy retrieval path below.

        # ── Legacy retrieval path (no augmented context, or empty for dim) ───
        relevant_chunks = _filter_chunks_for_dimension(chunks, dimension_id)
        sufficient, reason = check_evidence_sufficiency(relevant_chunks, dimension_id)

        if not sufficient and dimension_id not in max_reached:
            # Returning here exits run_analysis_agent immediately.  The orchestrator
            # will fetch more chunks, then call run_analysis_agent again from the top.
            # The resume check at the start of the loop ensures already-written
            # dimensions are skipped so work is never duplicated.
            query, filters = formulate_retrieval_query(dimension_id, facts, chunks)
            return RetrievalSignal(
                query=query,
                filters=filters,
                dimension=dimension_id,
            )

        # Evidence is sufficient (or max retries hit) — assess this dimension.
        prompt = build_dimension_prompt(
            facts, relevant_chunks, dimension_id, refined_context
        )
        # Wrap _call_llm with the reasoning decorator per-dimension so each LLM
        # call is logged with the correct dimension tag.  A new wrapper is created
        # on each iteration; the decorator is a lightweight closure and the cost
        # is negligible.  When no PrismLogger is active the wrapper is a no-op.
        _logged_llm = log_reasoning(agent="analysis", dimension=dimension_id)(_call_llm)
        response_text = _logged_llm(prompt, ANALYSIS_SYSTEM_PROMPT, llm_client)
        finding = parse_dimension_response(response_text, dimension_id)

        # Write immediately; do not batch.
        _write_dimension(memory, dimension_id, finding)

    return CompletionSignal(agent="analysis", message="All six dimensions assessed.")
