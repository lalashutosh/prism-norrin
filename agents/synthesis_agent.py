"""
agents/synthesis_agent.py
──────────────────────────
Synthesis agent: merges analysis and validation outputs into the twelve-section
final compliance report.

══════════════════════════════════════════════════════════════════════════════
INTELLIGENCE — pure functions, no I/O, fully testable without API calls
══════════════════════════════════════════════════════════════════════════════
  merge_analysis_with_validation(memory) -> dict
  build_agent_trace(facts, aug_ctx, merged, validation_flags, weak_claims, overturned) -> list
  build_evidence_separation(merged) -> dict
  build_synthesis_prompt(facts, merged, chunks, loop_count, ...) -> str
  parse_synthesis_response(response_text) -> (ReportSection, FollowUpSection, ConfidenceSection)
  determine_loop_condition(confidence_section, loop_count) -> bool
  _extract_json(text) -> dict
  _serialise_validation(memory) -> str
  _serialise_weak_claims(weak_claims) -> str
  _serialise_chunks_by_source(chunks) -> str

══════════════════════════════════════════════════════════════════════════════
ORCHESTRATION — coordinates calls, manages state, emits signals
══════════════════════════════════════════════════════════════════════════════
  _call_llm(prompt, system, client) -> str
  run_synthesis_agent(memory, chunks, context, llm_client) -> Signal
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional, Union

from core.types import (
    AugmentedContext,
    Chunk,
    ClaimStatus,
    CompletionSignal,
    Confidence,
    ConfidenceSection,
    DimensionFinding,
    FactSection,
    FollowUpSection,
    Label,
    LoopSignal,
    OverturnedClaim,
    ReportSection,
    ValidationSection,
    WeakClaim,
)
from core.logger import log_reasoning
from core.memory import SynthesisAgentMemoryView
from prompts.synthesis_prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT_TEMPLATE,
)
from agents.analysis_agent import _serialise_facts, _parse_confidence


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def merge_analysis_with_validation(
    memory: SynthesisAgentMemoryView,
) -> dict[str, Any]:
    """Build a merged view of the analysis findings incorporating validation.

    Merge rules:
      - OVERTURNED claims → replace original claim text/confidence/label with
        the validation finding and mark as UNCERTAIN in relevant dimensions.
      - Unresolved claim IDs (ValidationFlag.status == UNRESOLVED) → mark
        those claims' labels as UNCERTAIN in the merged output.
      - All other claims → carried through unchanged.

    Returns a plain dict keyed by dimension_id so the synthesis prompt can
    serialise it without importing agent-specific types.
    """
    # Build lookup: claim_id → OverturnedClaim
    overturned_by_id: dict[str, OverturnedClaim] = {
        oc.claim_id: oc for oc in memory.overturned_claims
    }
    # Build set of unresolved claim IDs
    unresolved_ids: set[str] = set()
    vf = memory.validation_flags
    if vf:
        for flag in vf.flags:
            if flag.status == ClaimStatus.UNRESOLVED:
                unresolved_ids.add(flag.claim_id)

    sections = {
        "definition_check":    memory.definition_check,
        "risk_classification": memory.risk_classification,
        "prohibited_practices": memory.prohibited_practices,
        "transparency":        memory.transparency,
        "roles":               memory.roles,
        "governance":          memory.governance,
    }

    merged: dict[str, Any] = {}
    for dim_id, section in sections.items():
        if section is None:
            merged[dim_id] = {
                "dimension_id": dim_id,
                "confidence":   Confidence.INSUFFICIENT.value,
                "summary":      "Section not available.",
                "claims":       [],
            }
            continue

        merged_claims = []
        for claim in section.claims:
            if claim.claim_id in overturned_by_id:
                oc = overturned_by_id[claim.claim_id]
                merged_claims.append({
                    "claim_id":   claim.claim_id,
                    "text":       oc.new_finding,
                    "label":      oc.new_label.value,
                    "confidence": oc.new_confidence.value,
                    "chunk_ids":  oc.new_chunk_ids,
                    "source":     "validation_override",
                })
            elif claim.claim_id in unresolved_ids:
                merged_claims.append({
                    "claim_id":   claim.claim_id,
                    "text":       claim.text,
                    "label":      Label.UNCERTAIN.value,
                    "confidence": Confidence.LOW.value,
                    "chunk_ids":  claim.chunk_ids,
                    "source":     "unresolved",
                })
            else:
                merged_claims.append({
                    "claim_id":   claim.claim_id,
                    "text":       claim.text,
                    "label":      claim.label.value,
                    "confidence": claim.confidence.value,
                    "chunk_ids":  claim.chunk_ids,
                    "source":     "analysis",
                })

        # Carry dimension-specific scalar fields.
        # getattr with a None default safely handles dimensions that don't
        # have a particular attribute (e.g. is_ai_system is only on DefinitionSection).
        # hasattr(val, "value") handles enum fields (RiskLevel, etc.) by serialising
        # them to their string value so the merged dict is plain JSON-serialisable.
        extra: dict = {}
        for attr in (
            "is_ai_system", "risk_level",
            "triggered_articles", "prohibited",
            "applies_to_gpai", "labelling_required", "notification_required",
            "is_provider", "is_deployer", "is_both",
            "documentation_required", "oversight_required", "monitoring_required",
        ):
            val = getattr(section, attr, None)
            if val is not None:
                extra[attr] = val.value if hasattr(val, "value") else val

        merged[dim_id] = {
            "dimension_id": dim_id,
            "confidence":   section.confidence.value,
            "summary":      section.summary,
            "claims":       merged_claims,
            **extra,
        }

    return merged


def build_agent_trace(
    facts: FactSection,
    aug_ctx: Optional[AugmentedContext],
    merged: dict[str, Any],
    validation_flags: Optional[ValidationSection],
    weak_claims: Optional[list[WeakClaim]],
    overturned_claims: Optional[list[OverturnedClaim]],
) -> list[dict]:
    """Build an ordered pipeline stage trace from actual execution data.

    Pure function — no LLM call.  Returns a list of stage dicts that faithfully
    reflect what each agent received and produced.  Because this is derived from
    real pipeline state (not generated by the LLM) it cannot hallucinate.

    Stages
    ------
    1. Document Extractor  — structured facts extracted from the uploaded doc.
    2. Legal Retriever     — EU AI Act provisions retrieved per dimension.
    3. Analysis Agent      — 6 dimensions assessed (in parallel when augmented).
    4. Validation Agent    — weak claims re-assessed with fresh evidence.
    5. Report Generator    — synthesis step currently running (status: in_progress).
    """
    trace: list[dict] = []

    # ── Stage 1: Document Extractor ───────────────────────────────────────────
    trace.append({
        "stage": 1,
        "agent": "Document Extractor",
        "description": (
            "Parsed the uploaded use-case document and extracted structured facts "
            "for downstream legal analysis."
        ),
        "status": "completed",
        "output": {
            "use_case_name":      facts.use_case_name,
            "industry":           facts.industry,
            "ai_capabilities":    facts.ai_capabilities,
            "data_inputs":        facts.data_inputs,
            "outputs":            facts.outputs,
            "deployment_context": facts.deployment_context,
            "affected_persons":   facts.affected_persons,
            "existing_oversight": facts.existing_oversight,
            "vendor_developer":   facts.vendor_or_developer,
            "source_chunk_count": len(facts.source_chunk_ids),
        },
    })

    # ── Stage 2: Legal Retriever (Augmentation) ───────────────────────────────
    aug_by_dim: dict[str, dict] = {}
    total_legal_chunks = 0
    if aug_ctx is not None:
        for dim, mappings in aug_ctx.mappings_by_dimension.items():
            leg_count  = sum(1 for m in mappings if m.source_type == "legislation")
            guid_count = sum(1 for m in mappings if m.source_type == "official_guidance")
            articles   = sorted({m.article_id for m in mappings if m.article_id})
            aug_by_dim[dim] = {
                "legislation_chunks":       leg_count,
                "official_guidance_chunks": guid_count,
                "article_ids":              articles,
            }
            total_legal_chunks += leg_count + guid_count

    trace.append({
        "stage": 2,
        "agent": "Legal Retriever",
        "description": (
            "Ran targeted retrieval for all six EU AI Act dimensions, mapping "
            "use-case facts to specific legislative provisions."
        ),
        "status": "completed",
        "output": {
            "dimensions_covered":  list(aug_by_dim.keys()),
            "total_legal_chunks":  total_legal_chunks,
            "user_chunks_loaded":  len(aug_ctx.user_chunks) if aug_ctx else 0,
            "by_dimension":        aug_by_dim,
        },
    })

    # ── Stage 3: Analysis Agent (parallel) ────────────────────────────────────
    analysis_by_dim: dict[str, dict] = {}
    for dim_id, dim_data in merged.items():
        conf   = dim_data.get("confidence", "INSUFFICIENT")
        summ   = dim_data.get("summary", "")
        n_clms = len(dim_data.get("claims", []))
        scalar: dict = {}
        for key in (
            "is_ai_system", "risk_level", "prohibited",
            "applies_to_gpai", "labelling_required", "notification_required",
            "is_provider", "is_deployer", "is_both",
            "documentation_required", "oversight_required", "monitoring_required",
        ):
            if key in dim_data:
                scalar[key] = dim_data[key]
        analysis_by_dim[dim_id] = {
            "confidence":   conf,
            "claims_count": n_clms,
            "summary":      summ[:200] + "…" if len(summ) > 200 else summ,
            **scalar,
        }

    trace.append({
        "stage": 3,
        "agent": "Analysis Agent (6 dimensions, parallel)",
        "description": (
            "Ran all six EU AI Act legal dimensions in parallel; each dimension "
            "received its own pre-filtered legislative chunks from the augmentation layer."
        ),
        "status": "completed",
        "output": analysis_by_dim,
    })

    # ── Stage 4: Validation Agent ─────────────────────────────────────────────
    weak_count      = len(weak_claims)    if weak_claims    else 0
    overturned_count = len(overturned_claims) if overturned_claims else 0
    confirmed_count = 0
    unresolved_count = 0
    if validation_flags:
        for flag in validation_flags.flags:
            if flag.status == ClaimStatus.CONFIRMED:
                confirmed_count += 1
            elif flag.status == ClaimStatus.UNRESOLVED:
                unresolved_count += 1

    validation_output: dict = {
        "weak_claims_identified": weak_count,
        "confirmed":              confirmed_count,
        "overturned":             overturned_count,
        "unresolved":             unresolved_count,
        "overall_confidence":     (
            validation_flags.overall_confidence.value if validation_flags
            else Confidence.INSUFFICIENT.value
        ),
    }
    if overturned_claims:
        validation_output["overturned_details"] = [
            {
                "claim_id":         oc.claim_id,
                "dimension":        oc.dimension_id,
                "original":         (
                    oc.original_claim_text[:150] + "…"
                    if len(oc.original_claim_text) > 150
                    else oc.original_claim_text
                ),
                "new_finding":      (
                    oc.new_finding[:150] + "…"
                    if len(oc.new_finding) > 150
                    else oc.new_finding
                ),
                "new_confidence":   oc.new_confidence.value,
                "new_label":        oc.new_label.value,
            }
            for oc in overturned_claims
        ]

    trace.append({
        "stage": 4,
        "agent": "Validation Agent",
        "description": (
            "Independently re-assessed weak and uncertain claims with fresh targeted "
            "retrieval; confirmed, overturned, or marked unresolved."
        ),
        "status": "completed",
        "output": validation_output,
    })

    # ── Stage 5: Report Generator (current) ───────────────────────────────────
    trace.append({
        "stage": 5,
        "agent": "Report Generator (Synthesis)",
        "description": (
            "Merges analysis and validation findings into the final twelve-section "
            "structured compliance report."
        ),
        "status": "in_progress",
        "output": {
            "action": "Synthesising final report from all pipeline outputs.",
        },
    })

    return trace


def build_evidence_separation(merged: dict[str, Any]) -> dict:
    """Group all claims from the merged analysis by epistemological label.

    Pure function — no LLM call.  Derived directly from the claim labels
    assigned during analysis and updated during validation.

    Returns a dict with keys FACT, RETRIEVED, ASSUMPTION, UNCERTAIN, each
    containing a list of claim dicts with their dimension and chunk citations.

    Useful for auditing the evidence quality of the final report at a glance:
      FACT       — supported by the client's own uploaded document.
      RETRIEVED  — grounded in legislation or official guidance corpus chunks.
      ASSUMPTION — inferred without direct evidence; clients should verify.
      UNCERTAIN  — could not be resolved even after validation retry.
    """
    groups: dict[str, list] = {
        "FACT":       [],
        "RETRIEVED":  [],
        "ASSUMPTION": [],
        "UNCERTAIN":  [],
    }
    for dim_id, dim_data in merged.items():
        for claim in dim_data.get("claims", []):
            label = str(claim.get("label", "UNCERTAIN")).upper()
            if label not in groups:
                label = "UNCERTAIN"
            groups[label].append({
                "claim_id":   claim.get("claim_id", ""),
                "dimension":  dim_id,
                "text":       claim.get("text", ""),
                "confidence": claim.get("confidence", ""),
                "chunk_ids":  claim.get("chunk_ids", []),
                "source":     claim.get("source", "analysis"),
            })
    return groups


def determine_loop_condition(
    confidence_section: ConfidenceSection,
    loop_count: int,
) -> bool:
    """Return True when the synthesis agent should trigger a re-analysis loop.

    Triggers when BOTH definition_check AND risk_classification are
    INSUFFICIENT and the global loop_count is still below the maximum (1).
    Critical dimensions failing simultaneously indicates the pipeline did not
    have enough evidence to make the most important findings.
    """
    if loop_count >= 1:
        return False  # max one loop; proceed with INSUFFICIENT markers
    return (
        confidence_section.definition_check == Confidence.INSUFFICIENT
        and confidence_section.risk_classification == Confidence.INSUFFICIENT
    )


def build_synthesis_prompt(
    facts: FactSection,
    merged: dict[str, Any],
    chunks: list[Chunk],
    loop_count: int,
    weak_claims: Optional[list[WeakClaim]] = None,
    validation_flags: Optional[ValidationSection] = None,
    agent_trace: Optional[list] = None,
) -> str:
    """Render the full synthesis prompt.

    Pure function — no I/O.

    Parameters
    ----------
    agent_trace : list | None
        Pre-built pipeline stage trace from build_agent_trace().  Injected
        into the prompt as AGENT TRACE context so the LLM can reference
        pipeline stages in the narrative summary.  Sections 11–12 of the
        ReportSection are populated programmatically from this same data
        after LLM parsing, not by the LLM itself.
    """
    loop_context = (
        f"This is loop pass {loop_count + 1}. "
        "Previous pass had insufficient evidence for critical dimensions. "
        "Be conservative and surface all remaining gaps explicitly in section 9."
        if loop_count > 0
        else ""
    )

    return SYNTHESIS_PROMPT_TEMPLATE.format(
        facts_text=_serialise_facts(facts),
        merged_analysis_json=json.dumps(merged, indent=2, default=str),
        validation_json=_serialise_validation_section(validation_flags),
        weak_claims_json=_serialise_weak_claims(weak_claims or []),
        agent_trace_json=json.dumps(agent_trace or [], indent=2, default=str),
        loop_context=loop_context,
        chunks_by_source_json=_serialise_chunks_by_source(chunks),
    )


def parse_synthesis_response(
    response_text: str,
) -> tuple[ReportSection, FollowUpSection, ConfidenceSection]:
    """Parse an LLM synthesis response into three typed output objects.

    Falls back to INSUFFICIENT / empty structures on any parse failure.
    Never raises.
    """
    raw = _extract_json(response_text)
    if not raw:
        return (
            ReportSection(),
            FollowUpSection(),
            ConfidenceSection(),   # all fields default to INSUFFICIENT
        )

    try:
        report_raw   = raw.get("report", {})
        follow_raw   = raw.get("follow_up", {})
        conf_raw     = raw.get("confidence", {})

        # confidence_score lives inside the report object so it is surfaced
        # as a named section in the output.  Fall back to the top-level
        # confidence dict if the LLM omits it from the report block.
        conf_score_raw = report_raw.get("confidence_score") or conf_raw

        report = ReportSection(
            use_case_summary=str(report_raw.get("use_case_summary", "")),
            extracted_facts=dict(report_raw.get("extracted_facts", {})),
            ai_definition_check=dict(report_raw.get("ai_definition_check", {})),
            risk_classification=dict(report_raw.get("risk_classification", {})),
            prohibited_practices_check=dict(
                report_raw.get("prohibited_practices_check", {})
            ),
            transparency_gpai_obligations=dict(
                report_raw.get("transparency_gpai_obligations", {})
            ),
            roles=dict(report_raw.get("roles", {})),
            governance_observations=dict(
                report_raw.get("governance_observations", {})
            ),
            missing_information=dict(report_raw.get("missing_information", {})),
            confidence_score=dict(conf_score_raw),
            citations_by_source=dict(report_raw.get("citations_by_source", {})),
            # evidence_separation and agent_trace are injected programmatically
            # in run_synthesis_agent() after this function returns.
        )

        follow_up = FollowUpSection(
            questions=list(follow_raw.get("questions", [])),
            missing_evidence=list(follow_raw.get("missing_evidence", [])),
        )

        confidence = ConfidenceSection(
            definition_check=_parse_confidence(conf_raw.get("definition_check", "INSUFFICIENT")),
            risk_classification=_parse_confidence(conf_raw.get("risk_classification", "INSUFFICIENT")),
            prohibited_practices=_parse_confidence(conf_raw.get("prohibited_practices", "INSUFFICIENT")),
            transparency=_parse_confidence(conf_raw.get("transparency", "INSUFFICIENT")),
            roles=_parse_confidence(conf_raw.get("roles", "INSUFFICIENT")),
            governance=_parse_confidence(conf_raw.get("governance", "INSUFFICIENT")),
            overall=_parse_confidence(conf_raw.get("overall", "INSUFFICIENT")),
        )

        return report, follow_up, confidence

    except Exception:  # noqa: BLE001
        return ReportSection(), FollowUpSection(), ConfidenceSection()


# ── Private intelligence helpers ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Three-strategy JSON extractor (same pattern as other agents)."""
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


def _serialise_validation_section(vf: Optional[ValidationSection]) -> str:
    """Serialise the ValidationSection to a JSON string for prompt injection.

    Each flag carries the per-claim verdict (CONFIRMED/OVERTURNED/UNRESOLVED)
    that the synthesis agent uses to apply overrides and mark uncertain items.
    """
    if vf is None:
        return "No validation data."
    return json.dumps(
        {
            "overall_confidence": vf.overall_confidence.value,
            "summary": vf.summary,
            "flags": [
                {
                    "claim_id":   f.claim_id,
                    "dimension":  f.dimension_id,
                    "status":     f.status.value,
                    "notes":      f.notes,
                }
                for f in vf.flags
            ],
        },
        indent=2,
    )


def _serialise_weak_claims(weak_claims: list[WeakClaim]) -> str:
    """Serialise weak claims for prompt injection.

    Included in the synthesis prompt so the model can surface the original
    weakness reasons in the final report's missing_information section.
    """
    if not weak_claims:
        return "[]"
    return json.dumps(
        [
            {
                "claim_id":   wc.claim_id,
                "dimension":  wc.dimension_id,
                "text":       wc.claim_text,
                "reason":     wc.reason,
            }
            for wc in weak_claims
        ],
        indent=2,
    )


def _serialise_chunks_by_source(chunks: list[Chunk]) -> str:
    """Group available chunks by source_type for the prompt's citation section.

    The synthesis agent uses this to populate citations_by_source in the final
    report — grouping by source makes it easy for the model to reference the
    correct chunk_ids under the correct source category.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_type].append(
            {"chunk_id": chunk.chunk_id, "article_id": chunk.article_id}
        )
    return json.dumps(dict(by_source), indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

@log_reasoning(agent="synthesis")
def _call_llm(prompt: str, system: str, client: Any) -> str:
    # dimension=None (default) because synthesis integrates all six dimensions
    # into one call rather than processing a single dimension.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=12000,   # increased for 12-section report with specific_obligations
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def run_synthesis_agent(
    memory: SynthesisAgentMemoryView,
    chunks: list[Chunk],
    context: dict,
    llm_client: Any = None,
) -> Union[LoopSignal, CompletionSignal]:
    """Drive the synthesis agent to produce the final compliance report.

    ORCHESTRATION ENTRY POINT.

    The agent:
      1. Merges analysis + validation into a consolidated view.
      2. Builds the synthesis prompt.
      3. Calls the LLM.
      4. Parses the response into (ReportSection, FollowUpSection, ConfidenceSection).
      5. Checks the loop condition on the PARSED confidence BEFORE writing.
         - If loop triggered: emits LoopSignal (no writes; memory stays clean).
         - If not: writes all three sections and emits CompletionSignal.

    context keys consumed:
      "loop_count" : int — current global loop count (max 1 before giving up)

    The loop condition check happens before any write so that if the synthesis
    determines critical dimensions are INSUFFICIENT the orchestrator can roll
    back to the after_analysis checkpoint without needing to undo any writes.
    """
    if llm_client is None:
        import anthropic
        llm_client = anthropic.Anthropic()

    facts = memory.facts
    if facts is None:
        raise ValueError("FactSection not available in memory.")

    loop_count: int = context.get("loop_count", 0)

    # -- INTELLIGENCE: merge and build prompt --
    merged = merge_analysis_with_validation(memory)

    # Build programmatic sections (agent trace + evidence separation).
    # These are constructed from real pipeline state — not LLM-generated —
    # so they accurately reflect what each agent received and produced.
    agent_trace = build_agent_trace(
        facts=facts,
        aug_ctx=memory.augmented_context,
        merged=merged,
        validation_flags=memory.validation_flags,
        weak_claims=memory.weak_claims,
        overturned_claims=memory.overturned_claims,
    )
    evidence_separation = build_evidence_separation(merged)

    prompt = build_synthesis_prompt(
        facts=facts,
        merged=merged,
        chunks=chunks,
        loop_count=loop_count,
        weak_claims=memory.weak_claims,
        validation_flags=memory.validation_flags,
        agent_trace=agent_trace,
    )

    # -- ORCHESTRATION: LLM call --
    response_text = _call_llm(prompt, SYNTHESIS_SYSTEM_PROMPT, llm_client)

    # -- INTELLIGENCE: parse --
    report, follow_up, confidence = parse_synthesis_response(response_text)

    # -- Loop condition check (BEFORE any write) --
    if determine_loop_condition(confidence, loop_count):
        refined = (
            follow_up.missing_evidence[0]
            if follow_up.missing_evidence
            else "Critical dimensions (AI system definition, risk classification) "
                 "lack sufficient legislative evidence. Retrieve more targeted chunks."
        )
        return LoopSignal(
            reason=(
                "Both definition_check and risk_classification are INSUFFICIENT. "
                "Looping back to analysis with refined context."
            ),
            refined_context=refined,
        )

    # -- Inject programmatic sections (Sections 11 & 12) into the parsed report --
    # Done after loop-check so we don't attach stale data to a discarded report.
    report.evidence_separation = evidence_separation
    report.agent_trace         = agent_trace
    # Mark Stage 5 as completed now that the report is being written.
    if report.agent_trace and report.agent_trace[-1].get("stage") == 5:
        report.agent_trace[-1]["status"] = "completed"
        report.agent_trace[-1]["output"]["action"] = "Final report written."

    # -- Write (only if not looping) --
    memory.write_final_report(report)
    memory.write_follow_up_questions(follow_up)
    memory.write_confidence_summary(confidence)

    return CompletionSignal(agent="synthesis", message="Final report written.")
