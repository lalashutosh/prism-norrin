"""
core/memory.py
──────────────
SessionMemory: single shared state for the entire pipeline.

Per-agent proxy views enforce write isolation and validate every write
across four checks (schema, citation integrity, label consistency,
confidence bounds).  Reads return deep copies so agents cannot mutate
shared state by modifying a returned object.  Sections an agent does not
own are not exposed at all — accessing them raises AttributeError, not a
permission error.

Nothing in this file triggers I/O, LLM calls, or retrieval.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional

from core.types import (
    AugmentedContext,
    Chunk,
    Claim,
    Confidence,
    ConfidenceSection,
    DefinitionSection,
    DimensionFinding,
    FactSection,
    FollowUpSection,
    GovernanceSection,
    Label,
    MemoryWriteError,
    OrchestratorState,
    OverturnedClaim,
    ProhibitedSection,
    ReportSection,
    RiskSection,
    RolesSection,
    TransparencySection,
    ValidationSection,
    WeakClaim,
)

# ── Source-type sets used by label-consistency validation ────────────────────

# Chunks from the built-in corpus — these may NOT be labeled FACT.
# Labeling a corpus chunk FACT would falsely imply the claim originated
# from the user's uploaded document rather than retrieved legislation.
CORPUS_SOURCE_TYPES: frozenset[str] = frozenset({
    "legislation",
    "official_guidance",
    "supporting_document",
    "case_law",
})

# Chunks from the user's uploaded document — these may NOT be labeled RETRIEVED.
# Labeling an uploaded chunk RETRIEVED would falsely imply it came from
# the legislation corpus rather than the user's own submission.
UPLOADED_SOURCE_TYPES: frozenset[str] = frozenset({
    "uploaded_doc",
})


# ── State-change logging helpers ─────────────────────────────────────────────
#
# These helpers let proxy write methods emit a StateChangeEntry via the active
# PrismLogger without importing logger.py at module level (deferred imports
# prevent any circular dependency since logger.py never imports memory.py).

def _section_to_dict(value: Any) -> Optional[dict]:
    """Convert a section value to a JSON-serializable dict.

    Handles:
      - dataclasses   → dataclasses.asdict() + JSON round-trip to convert enums
      - list          → wrapped as {"items": [...]} so the result is always a dict
      - None          → returned as None (used for previous_state on first write)
    The JSON round-trip converts enum members to their .value strings.
    """
    if value is None:
        return None
    try:
        if isinstance(value, list):
            # List fields (weak_claims, overturned_claims) are wrapped in a dict
            # so StateChangeEntry.new_state is always a plain dict, never a list.
            raw_items = [
                dataclasses.asdict(item) if dataclasses.is_dataclass(item) else item
                for item in value
            ]
            raw: Any = {"items": raw_items}
        elif dataclasses.is_dataclass(value):
            raw = dataclasses.asdict(value)
        else:
            raw = {"value": value}
        return json.loads(
            json.dumps(raw, default=lambda o: o.value if isinstance(o, enum.Enum) else str(o))
        )
    except (TypeError, ValueError):
        return {"error": "could not serialise section"}


def _log_state_change(
    agent: str,
    section: str,
    new_state: Optional[dict],
    previous_state: Optional[dict],
    write_validated: bool,
    validation_errors: Optional[list] = None,
) -> None:
    """Write a StateChangeEntry via the active PrismLogger, if one is active.

    Uses deferred imports from core.logger to avoid circular imports:
      core.logger → core.log_schema / core.log_store  (no import of memory.py)
      core.memory → core.logger  (deferred; executed only on first call)

    A no-op when no PrismLogger has been activated, so all agent unit tests
    that call proxy write methods directly remain completely unaffected.
    """
    from core.logger import _LOGGER_VAR, SESSION_ID_VAR      # noqa: PLC0415
    from core.log_schema import StateChangeEntry              # noqa: PLC0415

    active_logger = _LOGGER_VAR.get()
    session_id = SESSION_ID_VAR.get()

    if active_logger is None or not session_id:
        return  # no logger active — transparent no-op

    entry = StateChangeEntry(
        session_id=session_id,
        section=section,
        agent=agent,
        new_state=new_state or {},
        previous_state=previous_state,
        write_validated=write_validated,
        validation_errors=validation_errors or [],
    )
    active_logger.state_change(entry)


def _log_write(section_name: str, agent_name: str):
    """Decorator factory that adds StateChangeEntry logging to a proxy write method.

    Captures *previous_state* from ``self._memory.<section_name>`` before the
    write attempt, then logs the outcome after:
      - On success (validation passed, memory updated): write_validated=True.
      - On MemoryWriteError (validation failed): write_validated=False,
        validation_errors filled, exception re-raised unchanged.

    This keeps all logging logic in one place — the 13 write methods in the
    four proxy classes each get a one-line decorator instead of try/except blocks.

    Parameters
    ----------
    section_name : str
        The SessionMemory attribute name for this section (e.g. "definition_check").
    agent_name : str
        The owning agent name: "extraction" | "analysis" | "validation" | "synthesis".
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, value, *args, **kwargs):
            # Snapshot before the write so the log records what was there before.
            previous = getattr(self._memory, section_name, None)
            try:
                result = fn(self, value, *args, **kwargs)
                _log_state_change(
                    agent=agent_name,
                    section=section_name,
                    new_state=_section_to_dict(value),
                    previous_state=_section_to_dict(previous),
                    write_validated=True,
                )
                return result
            except MemoryWriteError as exc:
                # Log the failure before re-raising so the orchestrator still
                # receives the MemoryWriteError and can roll back.
                _log_state_change(
                    agent=agent_name,
                    section=section_name,
                    new_state=_section_to_dict(value),
                    previous_state=_section_to_dict(previous),
                    write_validated=False,
                    validation_errors=[f"{exc.check_name}: {exc.detail}"],
                )
                raise
        return wrapper
    return decorator


# ── SessionMemory ─────────────────────────────────────────────────────────────

@dataclass
class SessionMemory:
    """Single source of truth for the pipeline.

    Field ownership:
      facts                 → extraction agent writes, all agents read
      risk_classification … governance → analysis agent writes
      validation_flags … overturned_claims → validation agent writes
      final_report … confidence_summary  → synthesis agent writes
      _orchestrator         → orchestrator only; never exposed via proxies

    All fields start as None / empty so agents can detect whether a previous
    agent has completed its section (None = not yet written).
    """

    # Owned by extraction agent
    facts:                Optional[FactSection]       = None

    # Owned by analysis agent — one field per legal dimension.
    # Written sequentially; None means the dimension has not been assessed yet.
    risk_classification:  Optional[RiskSection]       = None
    definition_check:     Optional[DefinitionSection] = None
    prohibited_practices: Optional[ProhibitedSection] = None
    transparency:         Optional[TransparencySection] = None
    roles:                Optional[RolesSection]       = None
    governance:           Optional[GovernanceSection]  = None

    # Owned by validation agent
    validation_flags:     Optional[ValidationSection]  = None
    weak_claims:          list[WeakClaim]              = field(default_factory=list)
    overturned_claims:    list[OverturnedClaim]        = field(default_factory=list)

    # Owned by synthesis agent
    final_report:         Optional[ReportSection]      = None
    follow_up_questions:  Optional[FollowUpSection]    = None
    confidence_summary:   Optional[ConfidenceSection]  = None

    # Written by the orchestrator's augmentation phase (after extraction,
    # before analysis).  Read-only for all agents — they receive a deep copy
    # via their proxy view.  None until the augmentation phase completes.
    augmented_context:    Optional[AugmentedContext]   = None

    # Private: orchestrator only — never surfaced through any proxy.
    # Holds the retrieval cache, named checkpoints, retry counters, and
    # the set of all chunk IDs that have been retrieved this session.
    _orchestrator:        OrchestratorState            = field(
                              default_factory=OrchestratorState
                          )


# ── Write-gate helpers ────────────────────────────────────────────────────────
#
# MemoryWriteError is raised ONLY for type-level failures — values that
# downstream code cannot process correctly regardless of what any agent
# decides.  It is never raised for reasoning quality problems.
#
# Raises MemoryWriteError:
#   _check_schema       wrong Python type in a slot
#                       e.g. RiskSection where DefinitionSection expected →
#                       every downstream .is_ai_system access crashes
#
#   _check_confidence   raw string instead of Confidence enum
#   _check_label        raw string instead of Label enum
#                       e.g. "HIGH" instead of Confidence.HIGH →
#                       orchestrator loop condition silently always False
#
# Sanitises in-place (does NOT raise):
#   _sanitise_citations        hallucinated/unretrieved chunk_ids → strip +
#                              downgrade claim to ASSUMPTION/UNSUPPORTED so
#                              the validation agent reviews it
#
#   _sanitise_label_consistency  corpus chunk labeled FACT, or uploaded_doc
#                              chunk labeled RETRIEVED → correct the label
#                              in-place.  This is a quality mislabel, not a
#                              type failure — the pipeline can continue and
#                              the validation + synthesis chain surfaces it.
#
# Raises MemoryWriteError (verified-evidence writes only):
#   _check_citations    used in write_facts and write_overturned_claims where
#                       chunk_ids are pipeline-assigned, not LLM-generated.
#                       A non-retrieved ID there indicates a setup error.

def _check_schema(value: Any, expected_type: type, field_name: str) -> None:
    """Raise MemoryWriteError if value is not the expected dataclass type.

    Prevents passing the wrong section type to a write method
    (e.g. a RiskSection where a DefinitionSection is expected).
    This is a structural integrity check — the wrong type would cause
    AttributeError or silent data corruption in every downstream agent.
    """
    if not isinstance(value, expected_type):
        raise MemoryWriteError(
            "schema",
            f"{field_name}: expected {expected_type.__name__}, "
            f"got {type(value).__name__}",
        )


def _check_confidence(conf: Any, field_name: str) -> None:
    """Raise MemoryWriteError if conf is not a Confidence enum member.

    A raw string like "HIGH" would bypass enum comparison semantics,
    silently breaking the orchestrator's loop condition and quality
    regression checks.
    """
    if not isinstance(conf, Confidence):
        raise MemoryWriteError(
            "confidence_bounds",
            f"{field_name}: {conf!r} is not a Confidence enum member",
        )


def _check_label(label: Any, field_name: str) -> None:
    """Raise MemoryWriteError if label is not a Label enum member.

    A raw string would bypass label-consistency checks downstream.
    """
    if not isinstance(label, Label):
        raise MemoryWriteError(
            "confidence_bounds",
            f"{field_name}: {label!r} is not a Label enum member",
        )


def _check_citations(
    chunk_ids: list[str],
    retrieved_chunk_ids: set[str],
    field_name: str,
) -> None:
    """Raise MemoryWriteError if any chunk_id was never retrieved this session.

    This is a pipeline-setup check, not a quality check.  Call it only for
    agents writing verified evidence — extraction (document chunk_ids assigned
    by the ingest layer) and validation (new_chunk_ids from an explicit
    RetrievalSignal cycle).  A non-retrieved ID in those contexts means the
    orchestrator failed to register the chunk, not that the LLM hallucinated.

    Do NOT call from _validate_claims.  LLM-generated chunk_ids in analysis
    claims are sanitised by _sanitise_citations instead, so the section can
    still land and the validation agent can review it.
    """
    missing = [cid for cid in chunk_ids if cid not in retrieved_chunk_ids]
    if missing:
        raise MemoryWriteError(
            "schema",   # setup error: chunk was never registered in this session
            f"{field_name}: chunk_ids not registered in session: {missing}",
        )


def _sanitise_label_consistency(
    claim_or_overturn: Any,
    chunk_lookup: dict[str, Chunk],
    label_attr: str = "label",
    chunk_ids_attr: str = "chunk_ids",
) -> None:
    """Correct label/source-type mismatches in-place.

    A label mislabel is a reasoning quality problem, not a type failure:
    the Label value is a valid enum member — the LLM just picked the wrong
    one for the source.  The pipeline can continue; the validation agent will
    review the claim.  We correct the label so downstream audit trails are
    accurate without blocking the write.

    Corrections applied
    ~~~~~~~~~~~~~~~~~~~
    Corpus chunk (legislation / official_guidance) labeled FACT
        → correct to RETRIEVED
        Rationale: the evidence came from retrieved legislation, not the
        user's document.  FACT implies it was stated in the upload.

    Uploaded-doc chunk labeled RETRIEVED
        → correct to FACT
        Rationale: the evidence came from the user's document, not from
        retrieved legislation.  RETRIEVED implies a corpus source.

    Works on both Claim (label / chunk_ids) and OverturnedClaim
    (new_label / new_chunk_ids) via the attr arguments.
    """
    chunk_ids: list[str] = getattr(claim_or_overturn, chunk_ids_attr, [])
    label: Label         = getattr(claim_or_overturn, label_attr)

    for cid in chunk_ids:
        chunk = chunk_lookup.get(cid)
        if chunk is None:
            # Unknown in the current lookup — _sanitise_citations already
            # removed truly missing IDs.  Skip without raising.
            continue
        if chunk.source_type in CORPUS_SOURCE_TYPES and label == Label.FACT:
            setattr(claim_or_overturn, label_attr, Label.RETRIEVED)
            label = Label.RETRIEVED   # update local so subsequent chunks see the fix
        elif chunk.source_type in UPLOADED_SOURCE_TYPES and label == Label.RETRIEVED:
            setattr(claim_or_overturn, label_attr, Label.FACT)
            label = Label.FACT


def _validate_claims(
    claims: list[Claim],
    retrieved_chunk_ids: set[str],
    chunk_lookup: dict[str, Chunk],
    parent_name: str,
) -> None:
    """Gate structural integrity and sanitise quality issues for every claim.

    Per-claim processing order
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. _check_label / _check_confidence  (type gates — raise MemoryWriteError)
       Invalid enum instances would crash downstream comparisons silently.
       The section cannot land until these are fixed.

    2. _sanitise_citations               (quality sanitiser — mutates in-place)
       Strips hallucinated chunk_ids; downgrades affected claims to
       ASSUMPTION / UNSUPPORTED so the validation agent picks them up.

    3. _sanitise_label_consistency       (quality sanitiser — mutates in-place)
       Corrects FACT↔RETRIEVED mislabels based on the chunk's actual
       source type.  Both are valid Label enum values — the LLM just chose
       the wrong one.  Pipeline continues; audit trail is corrected.
    """
    for i, claim in enumerate(claims):
        loc = f"{parent_name}.claims[{i}]"
        _check_label(claim.label, f"{loc}.label")
        _check_confidence(claim.confidence, f"{loc}.confidence")
        _sanitise_citations(claim, retrieved_chunk_ids, loc)
        _sanitise_label_consistency(claim, chunk_lookup)


def _sanitise_citations(
    claim: Claim,
    retrieved_chunk_ids: set[str],
    loc: str,
) -> None:
    """Strip hallucinated chunk_ids from *claim* in-place.

    An LLM analysis agent may cite chunk IDs it never retrieved — either by
    hallucinating plausible-looking IDs or by referencing IDs from a prior
    session.  Rather than blocking the write (which would prevent the
    validation agent from ever seeing and correcting the claim), we:

      1. Remove the invalid IDs from claim.chunk_ids.
      2. If no valid citations remain, downgrade the claim to
         ASSUMPTION / LOW confidence and set is_weak = True / UNSUPPORTED.

    The degraded claim then flows through the normal validation path:
      identify_weak_claims picks it up (UNSUPPORTED reason) →
      validation agent requests targeted retrieval →
      confirm / overturn / UNRESOLVED →
      synthesis agent reports honest confidence.

    If some valid citations remain, the claim keeps its original label and
    confidence; the label-consistency check that follows will use the
    cleaned chunk_ids.
    """
    missing = [cid for cid in claim.chunk_ids if cid not in retrieved_chunk_ids]
    if not missing:
        return  # fast path — all citations are valid

    claim.chunk_ids = [cid for cid in claim.chunk_ids if cid in retrieved_chunk_ids]

    if not claim.chunk_ids:
        # No valid citations remain: the claim has no legislative grounding.
        # Mark as UNSUPPORTED so the validation agent treats it as a priority.
        claim.label       = Label.ASSUMPTION
        claim.confidence  = Confidence.LOW
        claim.is_weak     = True
        claim.weak_reason = "UNSUPPORTED"


def _validate_dimension_finding(
    finding: DimensionFinding,
    retrieved_chunk_ids: set[str],
    chunk_lookup: dict[str, Chunk],
    field_name: str,
) -> None:
    # Two-level validation: first check the section-level confidence,
    # then delegate to _validate_claims for per-claim checks.
    # Order matters: section confidence is checked before iterating claims
    # so that a missing enum value is caught immediately.
    _check_confidence(finding.confidence, f"{field_name}.confidence")
    _validate_claims(finding.claims, retrieved_chunk_ids, chunk_lookup, field_name)


# ── Proxy classes ─────────────────────────────────────────────────────────────
#
# Each proxy is constructed by the orchestrator and passed to the agent.
# The orchestrator supplies:
#   - a reference to the live SessionMemory object
#   - a reference to the live retrieved_chunk_ids set (mutated as the
#     orchestrator adds new chunks — the proxy always sees the current set)
#   - a fresh chunk_lookup dict built from all cached retrieval results
#     (rebuilt on every re-invocation so label checks use up-to-date metadata)
#
# Agents only interact with memory through these proxy objects.
# The orchestrator never passes the raw SessionMemory to an agent.

class ExtractionAgentMemoryView:
    """
    READ  : nothing (first agent; memory is empty at this point)
    WRITE : facts
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        # Use object.__setattr__ to bypass any potential __setattr__ override
        # and store the references directly on the instance dict.
        # This is the standard pattern for storing private state on proxy objects
        # that define __getattr__ — without it, self._memory = memory would
        # itself trigger __getattr__ before the attribute exists.
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    @_log_write("facts", "extraction")
    def write_facts(self, facts: FactSection) -> None:
        """Write the extraction output.

        Citation integrity for source_chunk_ids is checked only when the
        list is non-empty; the orchestrator must register document chunk IDs
        into retrieved_chunk_ids before calling this method.
        """
        _check_schema(facts, FactSection, "facts")
        # source_chunk_ids can legitimately be empty if the extraction agent
        # did not assign chunk IDs to document passages. Skip the citation
        # check in that case to avoid a false positive on an empty list.
        if facts.source_chunk_ids:
            _check_citations(
                facts.source_chunk_ids,
                self._retrieved_chunk_ids,
                "facts.source_chunk_ids",
            )
        # deepcopy prevents the caller from mutating memory by holding
        # a reference to the same FactSection object after writing.
        self._memory.facts = copy.deepcopy(facts)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal attribute lookup fails.
        # Because _memory, _retrieved_chunk_ids, and _chunk_lookup are set
        # via object.__setattr__ they ARE found by normal lookup and never
        # reach here. Any other name — including all of SessionMemory's public
        # fields — falls through to this sentinel and raises AttributeError.
        raise AttributeError(
            f"ExtractionAgentMemoryView does not expose attribute '{name}'"
        )


class AnalysisAgentMemoryView:
    """
    READ  : facts, validation_flags (loop-context on retry)
            + own section reads (for resume detection)
    WRITE : definition_check, risk_classification, prohibited_practices,
            transparency, roles, governance
    NOT EXPOSED: follow_up_questions, final_report, _orchestrator
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        # Same object.__setattr__ pattern as ExtractionAgentMemoryView —
        # necessary whenever __getattr__ is defined on the class.
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    # ── Readable (deepcopy) ──────────────────────────────────────────────────
    # All readable properties return a deepcopy so the agent cannot mutate
    # shared state by modifying the object it received.

    @property
    def facts(self) -> Optional[FactSection]:
        return copy.deepcopy(self._memory.facts)

    @property
    def augmented_context(self) -> Optional[AugmentedContext]:
        """Pre-computed user-doc → legal-provision mappings.

        Read-only.  Returns a deepcopy so the agent cannot mutate the shared
        context.  None until the orchestrator's augmentation phase completes.
        """
        return copy.deepcopy(self._memory.augmented_context)

    @property
    def validation_flags(self) -> Optional[ValidationSection]:
        # Exposed so the analysis agent can read any flags written by a previous
        # loop iteration — used as loop-context on retry after a LoopSignal.
        return copy.deepcopy(self._memory.validation_flags)

    # Own sections are readable so the agent can detect which dimensions it has
    # already written and skip them on re-invocation after a RetrievalSignal.
    # None means "not yet written"; a non-None value means "already done".
    @property
    def definition_check(self) -> Optional[DefinitionSection]:
        return copy.deepcopy(self._memory.definition_check)

    @property
    def risk_classification(self) -> Optional[RiskSection]:
        return copy.deepcopy(self._memory.risk_classification)

    @property
    def prohibited_practices(self) -> Optional[ProhibitedSection]:
        return copy.deepcopy(self._memory.prohibited_practices)

    @property
    def transparency(self) -> Optional[TransparencySection]:
        return copy.deepcopy(self._memory.transparency)

    @property
    def roles(self) -> Optional[RolesSection]:
        return copy.deepcopy(self._memory.roles)

    @property
    def governance(self) -> Optional[GovernanceSection]:
        return copy.deepcopy(self._memory.governance)

    # ── Writable (validated) ─────────────────────────────────────────────────
    # Every write method follows the same three-step pattern:
    #   1. _check_schema     — correct dataclass type?
    #   2. _validate_*       — all claims valid? citations exist? labels consistent?
    #   3. deepcopy-and-commit — write an independent copy into SessionMemory

    @_log_write("definition_check", "analysis")
    def write_definition(self, section: DefinitionSection) -> None:
        _check_schema(section, DefinitionSection, "definition_check")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "definition_check"
        )
        self._memory.definition_check = copy.deepcopy(section)

    @_log_write("risk_classification", "analysis")
    def write_risk(self, section: RiskSection) -> None:
        _check_schema(section, RiskSection, "risk_classification")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "risk_classification"
        )
        self._memory.risk_classification = copy.deepcopy(section)

    @_log_write("prohibited_practices", "analysis")
    def write_prohibited(self, section: ProhibitedSection) -> None:
        _check_schema(section, ProhibitedSection, "prohibited_practices")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "prohibited_practices"
        )
        self._memory.prohibited_practices = copy.deepcopy(section)

    @_log_write("transparency", "analysis")
    def write_transparency(self, section: TransparencySection) -> None:
        _check_schema(section, TransparencySection, "transparency")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "transparency"
        )
        self._memory.transparency = copy.deepcopy(section)

    @_log_write("roles", "analysis")
    def write_roles(self, section: RolesSection) -> None:
        _check_schema(section, RolesSection, "roles")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "roles"
        )
        self._memory.roles = copy.deepcopy(section)

    @_log_write("governance", "analysis")
    def write_governance(self, section: GovernanceSection) -> None:
        _check_schema(section, GovernanceSection, "governance")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "governance"
        )
        self._memory.governance = copy.deepcopy(section)

    def __getattr__(self, name: str) -> Any:
        # Catches any attribute access that isn't one of the explicitly defined
        # properties or methods above — including final_report, follow_up_questions,
        # and _orchestrator — and turns it into an AttributeError rather than
        # silently returning None or falling back to the underlying memory object.
        raise AttributeError(
            f"AnalysisAgentMemoryView does not expose attribute '{name}'"
        )


class ValidationAgentMemoryView:
    """
    READ  : facts, all six analysis sections
    WRITE : validation_flags, weak_claims, overturned_claims
    NOT EXPOSED: final_report, follow_up_questions, _orchestrator
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    # ── Readable ─────────────────────────────────────────────────────────────
    # The validation agent needs to read all six analysis sections to identify
    # which claims are weak and should be independently re-assessed.
    # All reads are deepcopy so the validation agent cannot accidentally
    # modify what the analysis agent wrote.

    @property
    def facts(self) -> Optional[FactSection]:
        return copy.deepcopy(self._memory.facts)

    @property
    def augmented_context(self) -> Optional[AugmentedContext]:
        """Pre-computed mappings — read-only for validation agent."""
        return copy.deepcopy(self._memory.augmented_context)

    @property
    def definition_check(self) -> Optional[DefinitionSection]:
        return copy.deepcopy(self._memory.definition_check)

    @property
    def risk_classification(self) -> Optional[RiskSection]:
        return copy.deepcopy(self._memory.risk_classification)

    @property
    def prohibited_practices(self) -> Optional[ProhibitedSection]:
        return copy.deepcopy(self._memory.prohibited_practices)

    @property
    def transparency(self) -> Optional[TransparencySection]:
        return copy.deepcopy(self._memory.transparency)

    @property
    def roles(self) -> Optional[RolesSection]:
        return copy.deepcopy(self._memory.roles)

    @property
    def governance(self) -> Optional[GovernanceSection]:
        return copy.deepcopy(self._memory.governance)

    # ── Writable ─────────────────────────────────────────────────────────────

    @_log_write("validation_flags", "validation")
    def write_validation_flags(self, section: ValidationSection) -> None:
        # ValidationSection has an overall_confidence field that must also
        # be checked; _validate_dimension_finding is not used here because
        # ValidationSection does not subclass DimensionFinding.
        _check_schema(section, ValidationSection, "validation_flags")
        _check_confidence(section.overall_confidence, "validation_flags.overall_confidence")
        self._memory.validation_flags = copy.deepcopy(section)

    @_log_write("weak_claims", "validation")
    def write_weak_claims(self, claims: list[WeakClaim]) -> None:
        # Validate the list itself before iterating, then validate each element.
        # WeakClaim does not carry chunk_ids (it references the original claim's
        # citations), so only schema, confidence, and label checks apply here.
        if not isinstance(claims, list):
            raise MemoryWriteError("schema", "weak_claims must be a list")
        for i, wc in enumerate(claims):
            _check_schema(wc, WeakClaim, f"weak_claims[{i}]")
            _check_confidence(wc.original_confidence, f"weak_claims[{i}].original_confidence")
            _check_label(wc.original_label, f"weak_claims[{i}].original_label")
        self._memory.weak_claims = copy.deepcopy(claims)

    @_log_write("overturned_claims", "validation")
    def write_overturned_claims(self, claims: list[OverturnedClaim]) -> None:
        # OverturnedClaim carries new_chunk_ids — the evidence the validation
        # agent explicitly retrieved to overturn the original claim.
        #
        # Type gates (MemoryWriteError):
        #   _check_schema / _check_confidence / _check_label — same as all writes.
        #   _check_citations — strict here because new_chunk_ids are pipeline-
        #   assigned during a RetrievalSignal cycle, not LLM-generated.  A
        #   non-retrieved ID indicates a setup error, not a quality issue.
        #
        # Quality sanitiser:
        #   _sanitise_label_consistency — corrects FACT↔RETRIEVED mislabels
        #   in-place using new_label / new_chunk_ids field names.
        if not isinstance(claims, list):
            raise MemoryWriteError("schema", "overturned_claims must be a list")
        for i, oc in enumerate(claims):
            _check_schema(oc, OverturnedClaim, f"overturned_claims[{i}]")
            _check_confidence(oc.new_confidence, f"overturned_claims[{i}].new_confidence")
            _check_label(oc.new_label, f"overturned_claims[{i}].new_label")
            _check_citations(
                oc.new_chunk_ids,
                self._retrieved_chunk_ids,
                f"overturned_claims[{i}]",
            )
            _sanitise_label_consistency(
                oc,
                self._chunk_lookup,
                label_attr="new_label",
                chunk_ids_attr="new_chunk_ids",
            )
        self._memory.overturned_claims = copy.deepcopy(claims)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"ValidationAgentMemoryView does not expose attribute '{name}'"
        )


class SynthesisAgentMemoryView:
    """
    READ  : facts, all six analysis sections, validation_flags,
            weak_claims, overturned_claims
    WRITE : final_report, follow_up_questions, confidence_summary
    NOT EXPOSED: _orchestrator
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    # ── Readable ─────────────────────────────────────────────────────────────
    # The synthesis agent has the broadest read access: it needs everything
    # produced by both the analysis and validation agents in order to merge
    # findings, apply overturned verdicts, and mark unresolved claims.

    @property
    def facts(self) -> Optional[FactSection]:
        return copy.deepcopy(self._memory.facts)

    @property
    def augmented_context(self) -> Optional[AugmentedContext]:
        """Pre-computed mappings — read-only for synthesis agent.

        The synthesis agent uses this to distinguish 'dimension UNRESOLVED
        because no corpus coverage exists' from 'dimension INSUFFICIENT
        because analysis never ran' — a distinction it should surface in
        the missing_information section.
        """
        return copy.deepcopy(self._memory.augmented_context)

    @property
    def definition_check(self) -> Optional[DefinitionSection]:
        return copy.deepcopy(self._memory.definition_check)

    @property
    def risk_classification(self) -> Optional[RiskSection]:
        return copy.deepcopy(self._memory.risk_classification)

    @property
    def prohibited_practices(self) -> Optional[ProhibitedSection]:
        return copy.deepcopy(self._memory.prohibited_practices)

    @property
    def transparency(self) -> Optional[TransparencySection]:
        return copy.deepcopy(self._memory.transparency)

    @property
    def roles(self) -> Optional[RolesSection]:
        return copy.deepcopy(self._memory.roles)

    @property
    def governance(self) -> Optional[GovernanceSection]:
        return copy.deepcopy(self._memory.governance)

    @property
    def validation_flags(self) -> Optional[ValidationSection]:
        return copy.deepcopy(self._memory.validation_flags)

    @property
    def weak_claims(self) -> list[WeakClaim]:
        return copy.deepcopy(self._memory.weak_claims)

    @property
    def overturned_claims(self) -> list[OverturnedClaim]:
        return copy.deepcopy(self._memory.overturned_claims)

    # ── Writable ─────────────────────────────────────────────────────────────

    @_log_write("final_report", "synthesis")
    def write_final_report(self, report: ReportSection) -> None:
        # ReportSection contains only plain dicts and strings, so no
        # citation or label checks are applied — schema alone is sufficient.
        _check_schema(report, ReportSection, "final_report")
        self._memory.final_report = copy.deepcopy(report)

    @_log_write("follow_up_questions", "synthesis")
    def write_follow_up_questions(self, section: FollowUpSection) -> None:
        _check_schema(section, FollowUpSection, "follow_up_questions")
        self._memory.follow_up_questions = copy.deepcopy(section)

    @_log_write("confidence_summary", "synthesis")
    def write_confidence_summary(self, section: ConfidenceSection) -> None:
        _check_schema(section, ConfidenceSection, "confidence_summary")
        # Iterate over every named confidence field in the section and validate
        # each one individually. getattr(section, fname) is used instead of
        # listing the values manually so that adding a new field to ConfidenceSection
        # only requires updating this tuple — not any surrounding logic.
        for fname in (
            "definition_check", "risk_classification", "prohibited_practices",
            "transparency", "roles", "governance", "overall",
        ):
            _check_confidence(
                getattr(section, fname),
                f"confidence_summary.{fname}",
            )
        self._memory.confidence_summary = copy.deepcopy(section)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"SynthesisAgentMemoryView does not expose attribute '{name}'"
        )
