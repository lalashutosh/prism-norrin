"""
core/orchestrator.py
─────────────────────
The orchestrator owns the full pipeline shape.  It is the only file that
knows the sequence of agents, manages checkpoints and rollbacks, handles all
RetrievalSignals, drives retry logic, and caches retrieval results.

This file contains NO prompts, NO legal reasoning, NO confidence scoring,
and NO output parsing.  All intelligence lives in the agent files.

Pipeline sequence
─────────────────
  1.  Initialise SessionMemory
  2.  Invoke extraction agent → save checkpoint "after_extraction"
  3.  Initial broad retrieve()  → populate retrieved_chunk_ids
  4.  Invoke analysis agent (signal loop, max 2 retries / dimension)
      → save checkpoint "after_analysis"
  5.  Invoke validation agent (signal loop, max 2 retries / claim)
      → save checkpoint "after_validation"
  6.  Invoke synthesis agent
      → LoopSignal  → rollback to "after_analysis", re-run analysis + validation
                       (max 1 global loop)
      → CompletionSignal → save checkpoint "after_synthesis"
  7.  Return memory.final_report to caller
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from typing import Any, Callable, Optional

from core.types import (
    Chunk,
    CompletionSignal,
    LoopSignal,
    MemoryWriteError,
    OrchestratorState,
    ReportSection,
    RetrievalSignal,
)
from core.memory import (
    AnalysisAgentMemoryView,
    ExtractionAgentMemoryView,
    SessionMemory,
    SynthesisAgentMemoryView,
    ValidationAgentMemoryView,
)
from agents.analysis_agent   import run_analysis_agent
from agents.validation_agent import run_validation_agent
from agents.synthesis_agent  import run_synthesis_agent
from core.log_schema import PipelineEvent, SignalEntry
from core.logger import PrismLogger

logger = logging.getLogger(__name__)

# Maximum retrieval retries per dimension (analysis) or claim (validation).
MAX_RETRIEVAL_RETRIES = 2
# Maximum global analysis+validation+synthesis loops.
MAX_LOOP_COUNT = 1


class Orchestrator:
    """Drives the full Prism compliance-analysis pipeline.

    Parameters
    ----------
    retrieve_fn : Callable[[str, dict], list[Chunk]]
        The RAG retrieval function.  Treated as a black box.
    extraction_fn : Callable[[str], FactSection] | None
        The extraction agent entry-point.  If None, the orchestrator expects
        FactSection to be injected directly via `run(facts=...)`.
    llm_client : optional
        Anthropic client.  None → agents create their own real clients.
        Pass a mock in tests.
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, dict], list[Chunk]],
        extraction_fn: Optional[Callable] = None,
        llm_client: Any = None,
        prism_logger: Optional[PrismLogger] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._retrieve_fn    = retrieve_fn
        self._extraction_fn  = extraction_fn
        self._llm_client     = llm_client
        # Optional structured logging.  None → all _log_* calls are no-ops so
        # existing tests that create Orchestrator without a logger continue to work.
        self._prism_logger   = prism_logger
        # Session ID for augmentation — used to load all user-doc chunks from
        # the session DB.  None → augmentation phase is silently skipped so
        # existing tests and in-memory document paths remain unaffected.
        self._session_id     = session_id

        self._memory: SessionMemory = SessionMemory()
        # Convenience alias — orchestrator accesses _orchestrator directly.
        self._state: OrchestratorState = self._memory._orchestrator

    # ── Logging helpers ──────────────────────────────────────────────────────
    # Both helpers are no-ops when no PrismLogger was provided, so the entire
    # logging layer is completely opt-in from the orchestrator's perspective.

    def _log_pipeline(
        self,
        event_type: str,
        agent: Optional[str] = None,
        duration_ms: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Emit one PipelineEvent to the active PrismLogger."""
        if self._prism_logger is None:
            return
        self._prism_logger.pipeline(PipelineEvent(
            session_id=self._prism_logger.session_id,
            event_type=event_type,
            agent=agent,
            duration_ms=duration_ms,
            metadata=metadata or {},
        ))

    def _log_signal(
        self,
        signal_type: str,
        agent: str,
        payload: dict,
        resolution: str,
        dimension: Optional[str] = None,
        retry_count: int = 0,
    ) -> Optional[int]:
        """Emit one SignalEntry to the active PrismLogger.

        Returns the database row ID so the caller can later call
        resolve_signal(id) to close the signal's lifecycle, or None when
        no logger is active.
        """
        if self._prism_logger is None:
            return None
        return self._prism_logger.signal(SignalEntry(
            session_id=self._prism_logger.session_id,
            signal_type=signal_type,
            agent=agent,
            payload=payload,
            resolution=resolution,
            dimension=dimension,
            retry_count=retry_count,
        ))

    # ── Public entry point ──────────────────────────────────────────────────

    def run(
        self,
        document_text: str = "",
        facts=None,  # FactSection | None — inject pre-extracted facts for testing
    ) -> ReportSection:
        """Execute the full pipeline and return the final compliance report.

        Parameters
        ----------
        document_text : str
            Raw text of the uploaded use-case document.  Passed to the
            extraction agent if one is configured.
        facts : FactSection | None
            If provided, skips the extraction phase and uses these facts
            directly.  Useful for testing and for external callers that
            already hold a FactSection.
        """
        session_start = time.time()
        self._log_pipeline("SESSION_STARTED", metadata={
            "document_text_length": len(document_text) if document_text else 0,
            "facts_injected": facts is not None,
        })

        # ── Step 1: extraction ──────────────────────────────────────────────
        if facts is not None:
            # Inject pre-extracted facts directly (bypass extraction proxy).
            # The orchestrator registers no doc chunk IDs, so source_chunk_ids
            # must be empty or pre-registered.
            self._memory.facts = copy.deepcopy(facts)
        elif self._extraction_fn is not None:
            self._run_extraction_phase(document_text)
        else:
            raise ValueError(
                "Either pass `facts=` directly or configure `extraction_fn`."
            )

        self._save_checkpoint("after_extraction")
        logger.info("Checkpoint saved: after_extraction")

        # ── Step 2: augmentation — map user-doc chunks to legal provisions ───
        # Runs before analysis so agents receive pre-filtered, dimension-specific
        # legal context.  Silently skipped when session_id is not provided (e.g.
        # tests that inject facts directly or use in-memory documents).
        self._run_augmentation_phase()
        logger.info("Augmentation phase complete.")

        # ── Step 3: initial broad retrieval (fallback / warm-up) ────────────
        # Still performed so the retrieval cache is populated and agents can
        # emit RetrievalSignals if the augmentation missed something.
        f = self._memory.facts
        initial_query = f"{f.use_case_name} {f.description}"[:300]
        self._retrieve_and_cache(initial_query, {})
        logger.info("Initial broad retrieval complete.")

        # ── Steps 4–6: analysis / validation / synthesis (with loop) ────────
        # loop_context carries the synthesis agent's refined guidance back into
        # the analysis agent on a second pass.  Empty on the first pass.
        loop_context: str = ""
        # range(MAX_LOOP_COUNT + 1) allows exactly one retry before giving up.
        for loop_iteration in range(MAX_LOOP_COUNT + 1):
            self._state.loop_count = loop_iteration

            self._run_analysis_phase(refined_context=loop_context)
            self._save_checkpoint("after_analysis")
            logger.info("Checkpoint saved: after_analysis (loop %d)", loop_iteration)

            self._run_validation_phase()
            self._save_checkpoint("after_validation")
            logger.info("Checkpoint saved: after_validation (loop %d)", loop_iteration)

            loop_signal = self._run_synthesis_phase()

            if loop_signal is None:
                # CompletionSignal — we are done.
                self._save_checkpoint("after_synthesis")
                logger.info("Checkpoint saved: after_synthesis")
                break

            # LoopSignal — roll back and re-run with refined context.
            if loop_iteration >= MAX_LOOP_COUNT:
                logger.warning(
                    "Max loops reached; proceeding with INSUFFICIENT markers."
                )
                # Force synthesis to complete on next attempt regardless.
                # The context flag is read by the synthesis agent via context dict.
                self._force_synthesis_complete()
                self._save_checkpoint("after_synthesis")
                break

            logger.info(
                "LoopSignal received: %s — rolling back to after_analysis.",
                loop_signal.reason,
            )
            loop_context = loop_signal.refined_context
            self._log_pipeline(
                "ROLLBACK_TRIGGERED",
                metadata={"reason": loop_signal.reason, "target_checkpoint": "after_analysis"},
            )
            self._restore_checkpoint("after_analysis")
            # Clear analysis + validation sections so the agents can re-write.
            self._reset_analysis_and_validation()

        self._log_pipeline(
            "SESSION_COMPLETED",
            duration_ms=int((time.time() - session_start) * 1000),
        )
        return self._memory.final_report

    # ── Extraction phase ─────────────────────────────────────────────────────

    def _run_extraction_phase(self, document_text: str) -> None:
        """Invoke the extraction function and write its output to memory.

        The extraction agent is external — its output (FactSection) is
        written directly to memory without going through a proxy, because
        extraction happens before retrieval so retrieved_chunk_ids is empty.
        """
        facts = self._extraction_fn(document_text)
        self._memory.facts = facts   # direct write; schema check omitted here
        # If the extraction function returned source_chunk_ids, register them.
        if facts.source_chunk_ids:
            for cid in facts.source_chunk_ids:
                self._state.retrieved_chunk_ids.add(cid)

    # ── Augmentation phase ────────────────────────────────────────────────────

    def _run_augmentation_phase(self) -> None:
        """Pre-compute user-document → legal-provision mappings for all dimensions.

        Loads ALL user-doc chunks from the session DB (not just the top-10 that
        retrieve() returns), then calls run_augmentation() which issues one
        targeted retrieve() per dimension with the correct source_type filter.

        Side effects:
        * memory.augmented_context is populated.
        * All legal chunk_ids from the mappings are registered in
          retrieved_chunk_ids so the memory proxy's citation sanitiser
          accepts claims that cite them.

        Silently no-ops when session_id is None (tests / in-memory paths).
        """
        if not self._session_id:
            logger.debug("No session_id — skipping augmentation phase.")
            return

        phase_start = time.time()
        self._log_pipeline("AGENT_STARTED", agent="augmentation")

        try:
            from retrieval.upload_ingest import load_session       # noqa: PLC0415
            from core.augmentation import run_augmentation         # noqa: PLC0415

            user_chunks, _ = load_session(self._session_id)
            if not user_chunks:
                logger.warning(
                    "Augmentation: no user chunks found for session '%s'.",
                    self._session_id,
                )
                self._log_pipeline(
                    "AGENT_COMPLETED", agent="augmentation",
                    duration_ms=int((time.time() - phase_start) * 1000),
                    metadata={"user_chunks": 0},
                )
                return

            augmented = run_augmentation(user_chunks, self._retrieve_fn)
            self._memory.augmented_context = augmented

            # Register every legal chunk_id as "retrieved" so the citation
            # sanitiser in the memory proxy accepts them as valid citations.
            total_mappings = 0
            aug_legal: list[Chunk] = []
            seen_legal: set[str] = set()
            for dim, mappings in augmented.mappings_by_dimension.items():
                for m in mappings:
                    if m.legal_chunk_id:
                        self._state.retrieved_chunk_ids.add(m.legal_chunk_id)
                        # Collect unique legal chunks for the retrieval cache (below).
                        if m.legal_chunk_id not in seen_legal:
                            seen_legal.add(m.legal_chunk_id)
                            aug_legal.append(Chunk(
                                chunk_id=m.legal_chunk_id,
                                text=m.legal_text,
                                source_type=m.source_type,
                                article_id=m.article_id or None,
                                metadata={},
                            ))
                total_mappings += len(mappings)

            # Also register user-doc chunk_ids so agents can cite FACT claims.
            for c in user_chunks:
                self._state.retrieved_chunk_ids.add(c.chunk_id)

            # ── Populate the retrieval cache with augmented chunks ────────────
            # run_augmentation() calls retrieve_fn() directly, bypassing
            # _retrieve_and_cache().  This means _get_all_chunks() — which only
            # returns chunks that went through the cache — does NOT contain the
            # targeted legislation chunks fetched per dimension.  Without them
            # the cache is dominated by Article 5 official_guidance from the
            # broad initial query, so the validation agent's legacy chunk-
            # selection path returns the wrong source for every non-prohibited
            # dimension.  Storing the augmented chunks under dedicated keys
            # makes them available to _get_all_chunks() as a correct fallback.
            if aug_legal:
                self._state.retrieval_cache["__augmentation_legal__"] = aug_legal
            if user_chunks:
                self._state.retrieval_cache["__augmentation_user__"] = list(user_chunks)

            self._log_pipeline(
                "AGENT_COMPLETED", agent="augmentation",
                duration_ms=int((time.time() - phase_start) * 1000),
                metadata={
                    "user_chunks": len(user_chunks),
                    "total_mappings": total_mappings,
                    "dimensions": list(augmented.mappings_by_dimension.keys()),
                },
            )
            logger.info(
                "Augmentation: %d user chunks mapped across %d dimensions "
                "(%d total mappings).",
                len(user_chunks),
                len(augmented.mappings_by_dimension),
                total_mappings,
            )

        except Exception as exc:   # noqa: BLE001
            # Augmentation failure must never crash the pipeline — fall back
            # to the legacy retrieve-on-demand path used by the analysis agent.
            logger.warning(
                "Augmentation phase failed (%s); continuing without it.", exc
            )
            self._log_pipeline(
                "AGENT_COMPLETED", agent="augmentation",
                duration_ms=int((time.time() - phase_start) * 1000),
                metadata={"error": str(exc)},
            )

    # ── Analysis phase ───────────────────────────────────────────────────────

    def _run_analysis_phase(self, refined_context: str = "") -> None:
        """Drive the analysis agent through its signal loop.

        Handles:
          - RetrievalSignal → cache-check → retrieve → re-invoke
          - MemoryWriteError → log and rollback if a critical section failed
          - retry_counts per dimension (max MAX_RETRIEVAL_RETRIES)
        """
        phase_start = time.time()
        self._log_pipeline("AGENT_STARTED", agent="analysis")

        context: dict = {
            "max_retrievals_reached": set(),
            "refined_context": refined_context,
        }
        # Clear per-dimension retry counts from any previous loop so a
        # dimension that ran out of retries in loop 0 gets fresh attempts
        # in loop 1 (if the synthesis agent triggers a second pass).
        for key in list(self._state.retry_counts.keys()):
            if key.startswith("analysis_"):
                del self._state.retry_counts[key]

        while True:
            view = self._make_analysis_view()
            try:
                signal = run_analysis_agent(
                    view, self._get_all_chunks(), context, self._llm_client
                )
            except MemoryWriteError as exc:
                self._handle_memory_write_error(exc, "after_extraction", "analysis")
                break

            if isinstance(signal, CompletionSignal):
                self._log_pipeline(
                    "AGENT_COMPLETED",
                    agent="analysis",
                    duration_ms=int((time.time() - phase_start) * 1000),
                )
                break
            if isinstance(signal, RetrievalSignal):
                retry_key = f"analysis_{signal.dimension}"
                count = self._state.retry_counts.get(retry_key, 0)
                if count >= MAX_RETRIEVAL_RETRIES:
                    logger.warning(
                        "Max retrieval retries for analysis dimension '%s'.",
                        signal.dimension,
                    )
                    context["max_retrievals_reached"].add(signal.dimension)
                    # Continue loop — agent will proceed with INSUFFICIENT.
                else:
                    sig_id = self._log_signal(
                        "RETRIEVAL_SIGNAL",
                        agent="analysis",
                        payload={"query": signal.query, "filters": signal.filters},
                        resolution="retrieving",
                        dimension=signal.dimension,
                        retry_count=count,
                    )
                    self._retrieve_and_cache(signal.query, signal.filters)
                    self._state.retry_counts[retry_key] = count + 1
                    if sig_id and self._prism_logger:
                        self._prism_logger.resolve_signal(sig_id)

    # ── Validation phase ─────────────────────────────────────────────────────

    def _run_validation_phase(self) -> None:
        """Drive the validation agent through its signal loop."""
        phase_start = time.time()
        self._log_pipeline("AGENT_STARTED", agent="validation")

        # The context dict is the validation agent's stateful scratchpad.
        # It is created here (not inside the agent) so it persists across
        # every re-invocation triggered by a RetrievalSignal.
        context: dict = {
            "processed_claim_ids":    set(),   # claim IDs already assessed; skip on re-entry
            "retry_counts":           {},       # per-claim retrieval retry counts
            "max_retrievals_reached": set(),    # claim IDs where retry limit was hit
            "overturned_claims":      [],       # accumulated OverturnedClaim objects
            "flags":                  [],       # accumulated ValidationFlag objects
            "unresolved_ids":         [],       # claim IDs with UNRESOLVED verdict
        }

        while True:
            view = self._make_validation_view()
            try:
                signal = run_validation_agent(
                    view, self._get_all_chunks(), context, self._llm_client
                )
            except MemoryWriteError as exc:
                self._handle_memory_write_error(exc, "after_analysis", "validation")
                break

            if isinstance(signal, CompletionSignal):
                self._log_pipeline(
                    "AGENT_COMPLETED",
                    agent="validation",
                    duration_ms=int((time.time() - phase_start) * 1000),
                )
                break
            if isinstance(signal, RetrievalSignal):
                claim_id = signal.dimension  # per spec, claim_id used as dimension
                retry_key = f"validation_{claim_id}"
                count = self._state.retry_counts.get(retry_key, 0)
                if count >= MAX_RETRIEVAL_RETRIES:
                    logger.warning(
                        "Max retrieval retries for validation claim '%s'.", claim_id
                    )
                    context["max_retrievals_reached"].add(claim_id)
                    # Update per-claim retry in context too.
                    context["retry_counts"][claim_id] = MAX_RETRIEVAL_RETRIES
                else:
                    sig_id = self._log_signal(
                        "RETRIEVAL_SIGNAL",
                        agent="validation",
                        payload={"query": signal.query, "filters": signal.filters},
                        resolution="retrieving",
                        dimension=claim_id,
                        retry_count=count,
                    )
                    self._retrieve_and_cache(signal.query, signal.filters)
                    self._state.retry_counts[retry_key] = count + 1
                    context["retry_counts"][claim_id] = count + 1
                    if sig_id and self._prism_logger:
                        self._prism_logger.resolve_signal(sig_id)

    # ── Synthesis phase ──────────────────────────────────────────────────────

    def _run_synthesis_phase(self) -> Optional[LoopSignal]:
        """Invoke the synthesis agent once.

        Returns None (CompletionSignal received) or the LoopSignal so the
        caller can decide whether to roll back.
        """
        phase_start = time.time()
        self._log_pipeline("AGENT_STARTED", agent="synthesis")

        context: dict = {"loop_count": self._state.loop_count}
        view = self._make_synthesis_view()
        try:
            signal = run_synthesis_agent(
                view, self._get_all_chunks(), context, self._llm_client
            )
        except MemoryWriteError as exc:
            self._handle_memory_write_error(exc, "after_validation", "synthesis")
            return None

        if isinstance(signal, CompletionSignal):
            # Quality-regression safety net.
            if self._check_quality_regression():
                logger.warning(
                    "Quality regression detected after synthesis; "
                    "proceeding anyway (loop limit)."
                )
            self._log_pipeline(
                "AGENT_COMPLETED",
                agent="synthesis",
                duration_ms=int((time.time() - phase_start) * 1000),
            )
            return None

        if isinstance(signal, LoopSignal):
            self._log_pipeline("LOOP_TRIGGERED", metadata={"reason": signal.reason})
            self._log_signal(
                "LOOP_SIGNAL",
                agent="synthesis",
                payload={"reason": signal.reason, "refined_context": signal.refined_context},
                resolution="rolling_back_to_after_analysis",
            )
            return signal

        raise ValueError(f"Unexpected signal from synthesis agent: {type(signal)}")

    def _force_synthesis_complete(self) -> None:
        """Run synthesis one final time ignoring the loop condition.

        Called when loop_count has reached MAX_LOOP_COUNT so that we always
        produce a final report (potentially with INSUFFICIENT markers).
        """
        # Increment loop_count beyond max so determine_loop_condition returns False.
        self._state.loop_count = MAX_LOOP_COUNT + 1
        context: dict = {"loop_count": self._state.loop_count}
        view = self._make_synthesis_view()
        try:
            run_synthesis_agent(view, self._get_all_chunks(), context, self._llm_client)
        except MemoryWriteError as exc:
            logger.error("MemoryWriteError in forced synthesis: %s", exc)

    # ── Checkpointing ────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str) -> None:
        """Deep-copy the data sections of SessionMemory as a named checkpoint.

        Checkpoints exclude _orchestrator.checkpoints itself to prevent
        recursive deep copies.
        """
        snapshot = copy.deepcopy(self._memory)
        # Clear nested checkpoints in the snapshot to avoid infinite nesting.
        snapshot._orchestrator.checkpoints = {}
        self._state.checkpoints[name] = snapshot
        logger.debug("Checkpoint '%s' saved.", name)
        self._log_pipeline("CHECKPOINT_SAVED", metadata={"checkpoint": name})

    def _restore_checkpoint(self, name: str) -> None:
        """Restore SessionMemory data sections from a named checkpoint.

        Preserves the orchestrator's infrastructure (cache, chunk_ids, counts,
        checkpoints) across rollbacks so we don't re-retrieve already-fetched
        chunks.
        """
        snapshot = self._state.checkpoints.get(name)
        if snapshot is None:
            raise KeyError(f"No checkpoint named '{name}'")

        # Restore only data sections.
        self._memory.facts                = copy.deepcopy(snapshot.facts)
        self._memory.risk_classification  = copy.deepcopy(snapshot.risk_classification)
        self._memory.definition_check     = copy.deepcopy(snapshot.definition_check)
        self._memory.prohibited_practices = copy.deepcopy(snapshot.prohibited_practices)
        self._memory.transparency         = copy.deepcopy(snapshot.transparency)
        self._memory.roles                = copy.deepcopy(snapshot.roles)
        self._memory.governance           = copy.deepcopy(snapshot.governance)
        self._memory.validation_flags     = copy.deepcopy(snapshot.validation_flags)
        self._memory.weak_claims          = copy.deepcopy(snapshot.weak_claims)
        self._memory.overturned_claims    = copy.deepcopy(snapshot.overturned_claims)
        self._memory.final_report         = copy.deepcopy(snapshot.final_report)
        self._memory.follow_up_questions  = copy.deepcopy(snapshot.follow_up_questions)
        self._memory.confidence_summary   = copy.deepcopy(snapshot.confidence_summary)
        # augmented_context is set once during the augmentation phase (before
        # analysis) and never changes during analysis/validation/synthesis.
        # Restoring it explicitly here is a safety net: it ensures the correct
        # pre-mapped context is always available after a loop rollback, even if
        # a future refactor modifies when augmented_context is written.
        self._memory.augmented_context    = copy.deepcopy(snapshot.augmented_context)
        logger.debug("Checkpoint '%s' restored.", name)
        self._log_pipeline("CHECKPOINT_RESTORED", metadata={"checkpoint": name})

    def _reset_analysis_and_validation(self) -> None:
        """Clear analysis + validation sections so agents can rewrite them."""
        self._memory.risk_classification  = None
        self._memory.definition_check     = None
        self._memory.prohibited_practices = None
        self._memory.transparency         = None
        self._memory.roles                = None
        self._memory.governance           = None
        self._memory.validation_flags     = None
        self._memory.weak_claims          = []
        self._memory.overturned_claims    = []

    # ── Retrieval + KV cache ─────────────────────────────────────────────────

    def _retrieve_and_cache(self, query: str, filters: dict) -> list[Chunk]:
        """Cache-check → retrieve → extend chunk pool.

        The cache key is a deterministic hash of (query, sorted filters).
        Returns cached result without calling retrieve() if present.
        """
        cache_key = _make_cache_key(query, filters)
        if cache_key in self._state.retrieval_cache:
            logger.debug("Cache hit for query: %s", query[:60])
            self._log_pipeline("CACHE_HIT", metadata={"query": query[:200], "filters": filters})
            return self._state.retrieval_cache[cache_key]

        self._log_pipeline("RETRIEVE_CALLED", metadata={"query": query[:200], "filters": filters})
        chunks = self._retrieve_fn(query, filters)
        self._state.retrieval_cache[cache_key] = chunks
        for chunk in chunks:
            self._state.retrieved_chunk_ids.add(chunk.chunk_id)
        logger.debug(
            "Retrieved %d chunk(s) for query: %s", len(chunks), query[:60]
        )
        return chunks

    # ── Proxy factory methods ────────────────────────────────────────────────

    def _make_analysis_view(self) -> AnalysisAgentMemoryView:
        return AnalysisAgentMemoryView(
            self._memory,
            self._state.retrieved_chunk_ids,
            self._build_chunk_lookup(),
        )

    def _make_validation_view(self) -> ValidationAgentMemoryView:
        return ValidationAgentMemoryView(
            self._memory,
            self._state.retrieved_chunk_ids,
            self._build_chunk_lookup(),
        )

    def _make_synthesis_view(self) -> SynthesisAgentMemoryView:
        return SynthesisAgentMemoryView(
            self._memory,
            self._state.retrieved_chunk_ids,
            self._build_chunk_lookup(),
        )

    # ── Chunk pool helpers ───────────────────────────────────────────────────

    def _build_chunk_lookup(self) -> dict[str, Chunk]:
        """Build {chunk_id: Chunk} from all cached retrieval results."""
        lookup: dict[str, Chunk] = {}
        for chunks in self._state.retrieval_cache.values():
            for chunk in chunks:
                lookup[chunk.chunk_id] = chunk
        return lookup

    def _get_all_chunks(self) -> list[Chunk]:
        """Return a flat deduplicated list of all retrieved chunks."""
        seen: set[str] = set()
        result: list[Chunk] = []
        for chunks in self._state.retrieval_cache.values():
            for chunk in chunks:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    result.append(chunk)
        return result

    # ── Quality regression check ─────────────────────────────────────────────

    def _check_quality_regression(self) -> bool:
        """Return True when synthesis confidence is suspiciously low.

        Triggers when BOTH definition_check AND risk_classification are
        INSUFFICIENT in the written confidence_summary, indicating the
        synthesis agent could not make the two most fundamental findings.
        """
        cs = self._memory.confidence_summary
        if cs is None:
            return False
        from core.types import Confidence
        return (
            cs.definition_check == Confidence.INSUFFICIENT
            and cs.risk_classification == Confidence.INSUFFICIENT
        )

    # ── Error handling ───────────────────────────────────────────────────────

    def _handle_memory_write_error(
        self,
        error: MemoryWriteError,
        fallback_checkpoint: str,
        agent_name: str,
    ) -> None:
        """Roll back session state on structural type failures.

        MemoryWriteError fires only on type-level corruption — values that
        downstream pipeline code cannot process correctly regardless of what
        any agent later decides:
          - Wrong dataclass type in a slot ("schema")
          - Raw string instead of Confidence/Label enum ("confidence_bounds")

        Everything else is sanitised upstream and never reaches here:
          - Hallucinated chunk_ids   → _sanitise_citations (strip + downgrade)
          - Label/source mislabels   → _sanitise_label_consistency (correct in-place)
          - Low confidence / weak claims → handled by validation + synthesis

        The orchestrator has no EU AI Act context and cannot judge reasoning
        quality.  Its only role here is to detect that a section slot contains
        mechanically unusable data and roll the pipeline back to a known-good
        checkpoint so the producing agent can retry.
        """
        logger.error(
            "MemoryWriteError in %s agent [%s]: %s",
            agent_name,
            error.check_name,
            error.detail,
        )
        self._log_signal(
            "MEMORY_WRITE_ERROR",
            agent=agent_name,
            payload={"check_name": error.check_name, "detail": error.detail},
            resolution=f"rolling_back_to_{fallback_checkpoint}",
        )
        try:
            self._restore_checkpoint(fallback_checkpoint)
            logger.warning("Rolled back to checkpoint '%s'.", fallback_checkpoint)
        except KeyError:
            logger.warning(
                "No checkpoint '%s' available for rollback.", fallback_checkpoint
            )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_cache_key(query: str, filters: dict) -> str:
    """Deterministic hash of (query, sorted filters) for the retrieval cache."""
    payload = query + json.dumps(sorted(filters.items()), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
