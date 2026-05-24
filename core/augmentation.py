"""
core/augmentation.py
─────────────────────
Augmentation layer: maps every user-document chunk to relevant EU AI Act
legal provisions before analysis agents run.

This eliminates two root-cause problems in the analysis pipeline:

1. CITATION HALLUCINATION
   Analysis agents previously received a large, unfiltered chunk pool and
   invented chunk_ids they never retrieved.  With the augmentation layer,
   each agent receives only the pre-mapped chunk_ids for its dimension —
   the prompt explicitly lists the only valid citations.

2. RETRIEVAL DILUTION
   The Article 5 official-guidance corpus (99 nodes) dominated retrieval
   for every dimension because generic queries matched it by volume.
   The augmentation uses dedicated per-dimension queries so each dimension
   gets legislation-specific context, not an Article-5-heavy mix.

Pipeline position
─────────────────
  Extraction  →  [Augmentation]  →  Analysis  →  Validation  →  Synthesis
                     ↑
   reads ALL user-doc chunks from the session DB
   runs one targeted retrieve() call per dimension
   maps every user chunk to matching legal provisions
   stores AugmentedContext in SessionMemory

══════════════════════════════════════════════════════════════════════════════
PUBLIC API
══════════════════════════════════════════════════════════════════════════════
  run_augmentation(user_chunks, retrieve_fn)       → AugmentedContext
  build_dimension_chunks(ctx, dimension_id)         → list[Chunk]
  serialise_augmented_context(ctx, dimension_id)   → str
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

from core.types import AugmentedContext, Chunk, FactToLawMapping

# ── Per-dimension retrieval configuration ────────────────────────────────────
#
# Each entry: (query_string, source_types_list)
# The query is sent to retrieve() with the given source_types filter.
# "legislation" — fetches from the EU AI Act Regulation corpus.
# "official_guidance" — fetches from the Article 5 guidelines corpus.
#
# Rule: prohibited_practices is the ONLY dimension that should pull
# official_guidance.  All others use legislation exclusively to prevent
# Article 5 guidance from diluting non-prohibited-practices dimensions.

DIMENSION_RETRIEVAL_CONFIG: dict[str, tuple[str, list[str]]] = {
    "definition_check": (
        "AI system definition Article 3 machine learning deep learning Annex I technique",
        ["legislation"],
    ),
    "risk_classification": (
        "high risk AI system classification Article 6 Annex III categories employment education",
        ["legislation"],
    ),
    "prohibited_practices": (
        "prohibited practices Article 5 manipulation social scoring biometric identification",
        ["legislation", "official_guidance"],
    ),
    "transparency": (
        "transparency obligations Article 13 Article 50 inform deployers natural persons disclosure",
        ["legislation"],
    ),
    "roles": (
        "provider deployer definition Article 3 responsibilities Article 25 Article 26",
        ["legislation"],
    ),
    "governance": (
        "governance risk management technical documentation Article 9 10 11 17 high risk obligations",
        ["legislation"],
    ),
}

# Minimum keyword-overlap score to create a user→legal mapping.
# 0.05 = at least ~5 % of user-chunk tokens appear in the legal text.
# Set intentionally low so even sparse overlap creates a mapping entry;
# agents receive the mapping and decide relevance themselves.
_MIN_OVERLAP = 0.05

# Maximum legal chunks retained per dimension (before user-chunk mapping).
# retrieve() already caps at _TOP_K_RESULTS=10 so this is an extra guard.
_MAX_LEGAL_PER_DIM = 10

# Tokeniser: lowercase alphanumeric tokens only.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_augmentation(
    user_chunks: list[Chunk],
    retrieve_fn: Callable[[str, dict], list[Chunk]],
) -> AugmentedContext:
    """Map every user-document chunk to relevant legal provisions per dimension.

    For each dimension:
      1. Run a targeted retrieve() call with dimension-specific query +
         source_types filter.  This guarantees the right corpus is queried
         and Article 5 guidance does not bleed into non-prohibited dimensions.
      2. For each (user_chunk, legal_chunk) pair, compute keyword overlap.
         Pairs above _MIN_OVERLAP become FactToLawMapping entries.
      3. If no overlap at all, include all legal chunks without a user mapping
         ("unanchored" entries) — the dimension still gets legal context even
         if the user document doesn't share vocabulary with the legislation.

    Parameters
    ──────────
    user_chunks   All chunks loaded from the user's session DB.
    retrieve_fn   The same retrieve_fn the orchestrator uses; must already
                  have session_id baked in via functools.partial.

    Returns
    ───────
    AugmentedContext with mappings_by_dimension populated for all 6 dimensions
    and user_chunks set to the full user-document chunk list.
    """
    mappings_by_dimension: dict[str, list[FactToLawMapping]] = {}

    for dimension, (query, source_types) in DIMENSION_RETRIEVAL_CONFIG.items():
        legal_chunks = retrieve_fn(
            query,
            {"source_types": source_types, "dimension": dimension},
        )[:_MAX_LEGAL_PER_DIM]

        if not legal_chunks:
            mappings_by_dimension[dimension] = []
            continue

        dim_mappings = _map_chunks(user_chunks, legal_chunks, dimension)
        mappings_by_dimension[dimension] = dim_mappings

    return AugmentedContext(
        mappings_by_dimension=mappings_by_dimension,
        user_chunks=user_chunks,
    )


def build_dimension_chunks(
    ctx: AugmentedContext,
    dimension_id: str,
) -> list[Chunk]:
    """Return deduplicated Chunk objects for all legal provisions in *dimension_id*.

    Reconstructs Chunk objects from FactToLawMapping entries.  The returned
    list contains only legal corpus chunks — user-doc chunks are accessible
    separately via ctx.user_chunks.

    Used by the analysis agent to build the ``chunks`` argument that the
    evidence-sufficiency check and prompt serialiser consume.
    """
    mappings = ctx.mappings_by_dimension.get(dimension_id, [])
    seen: set[str] = set()
    chunks: list[Chunk] = []

    for m in mappings:
        if not m.legal_chunk_id or m.legal_chunk_id in seen:
            continue
        seen.add(m.legal_chunk_id)
        chunks.append(Chunk(
            chunk_id    = m.legal_chunk_id,
            text        = m.legal_text,
            source_type = m.source_type,
            article_id  = m.article_id or None,
            metadata    = {},
        ))

    return chunks


def serialise_augmented_context(
    ctx: AugmentedContext,
    dimension_id: str,
) -> str:
    """Render pre-mapped context for injection into an analysis prompt.

    Output format:

        PRE-MAPPED LEGAL CONTEXT  (cite ONLY chunk_ids listed here)
        ─────────────────────────────────────────────────────────────
        [1] chunk_id=AIA_Art3_1_ai_system_definition  [Article 3(1)]  [legislation]
            'AI system' means a machine-based system...
            ← User fact: "The system uses an LSTM neural network..."

        [2] chunk_id=AIA_AnnexI_ai_techniques  [Annex I]  [legislation]
            Machine learning approaches, including deep learning...
            ← User fact: "The system uses an LSTM neural network..."

        USER-DOCUMENT CHUNKS  (cite these for FACT claims)
        ─────────────────────────────────────────────────────────────
        [A] chunk_id=doc_chunk_003
            The system uses an LSTM neural network trained on vibration sensor data...

    When dimension_id has no mappings, returns an empty string so the caller
    can fall back to the standard chunk serialiser.
    """
    mappings = ctx.mappings_by_dimension.get(dimension_id, [])
    if not mappings:
        return ""

    # ── Legal chunks section ─────────────────────────────────────────────────
    # Group by legal_chunk_id to avoid repeating the same provision; collect
    # user facts that matched each legal chunk.
    legal_order: list[str] = []
    legal_map: dict[str, FactToLawMapping] = {}
    user_facts_for: dict[str, list[str]] = defaultdict(list)

    for m in mappings:
        if m.legal_chunk_id not in legal_map:
            legal_order.append(m.legal_chunk_id)
            legal_map[m.legal_chunk_id] = m
        if m.user_chunk_text:
            # Deduplicate user facts shown under a legal chunk.
            existing = user_facts_for[m.legal_chunk_id]
            snippet = m.user_chunk_text[:120].strip()
            if snippet and snippet not in existing:
                existing.append(snippet)

    legal_parts: list[str] = []
    for i, chunk_id in enumerate(legal_order, 1):
        m = legal_map[chunk_id]
        header = (
            f"[{i}] chunk_id={chunk_id}"
            + (f"  [{m.article_id}]" if m.article_id else "")
            + f"  [{m.source_type}]"
        )
        body = m.legal_text.strip()
        user_facts = user_facts_for.get(chunk_id, [])
        fact_lines = (
            "\n".join(f"    ← User fact: \"{uf}…\"" for uf in user_facts[:2])
            if user_facts else ""
        )
        legal_parts.append(f"{header}\n{body}" + (f"\n{fact_lines}" if fact_lines else ""))

    legal_section = (
        "PRE-MAPPED LEGAL CONTEXT  (cite ONLY chunk_ids listed here)\n"
        + "─" * 65 + "\n"
        + "\n\n".join(legal_parts)
    )

    # ── User-document chunks section ─────────────────────────────────────────
    # Collect the user chunks that participated in at least one mapping for
    # this dimension; also add any other user chunks so the agent can always
    # cite FActs directly from the document.
    mapped_user_ids: set[str] = {
        m.user_chunk_id for m in mappings if m.user_chunk_id
    }
    # Prefer mapped chunks first; pad with remaining user_chunks up to 8 total.
    ordered_user: list[Chunk] = []
    user_by_id = {c.chunk_id: c for c in ctx.user_chunks}

    for uid in mapped_user_ids:
        if uid in user_by_id:
            ordered_user.append(user_by_id[uid])

    for c in ctx.user_chunks:
        if c.chunk_id not in mapped_user_ids:
            ordered_user.append(c)
        if len(ordered_user) >= 8:
            break

    user_parts: list[str] = []
    for i, chunk in enumerate(ordered_user, 0):
        label = chr(ord("A") + i) if i < 26 else str(i)
        snippet = chunk.text[:300].strip()
        user_parts.append(f"[{label}] chunk_id={chunk.chunk_id}\n    {snippet}")

    user_section = (
        "USER-DOCUMENT CHUNKS  (cite these for FACT claims, label=FACT)\n"
        + "─" * 65 + "\n"
        + "\n\n".join(user_parts)
        if user_parts else ""
    )

    return "\n\n".join(filter(None, [legal_section, user_section]))


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> set[str]:
    """Return the set of lowercase alphanumeric tokens in *text*."""
    return set(_TOKEN_RE.findall(text.lower()))


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Keyword overlap score: |tokens(a) ∩ tokens(b)| / |tokens(a)|.

    Returns 0.0 for empty inputs or when text_a has no tokens.
    Used to score how much of the user chunk's vocabulary appears in the
    legal provision, giving a rough estimate of topical relevance.
    """
    if not text_a or not text_b:
        return 0.0
    ta = _tokenize(text_a)
    if not ta:
        return 0.0
    tb = _tokenize(text_b)
    return len(ta & tb) / len(ta)


def _map_chunks(
    user_chunks: list[Chunk],
    legal_chunks: list[Chunk],
    dimension: str,
) -> list[FactToLawMapping]:
    """Create FactToLawMapping entries for relevant (user, legal) pairs.

    Strategy
    ────────
    1. For each user chunk, score it against every legal chunk.
    2. Keep pairs whose keyword overlap ≥ _MIN_OVERLAP.
    3. If nothing cleared the threshold, fall back to including all legal
       chunks as "unanchored" mappings (user_chunk_id="") so the dimension
       is never starved of legal context.

    Ordering: mappings are sorted legal-chunk-first (to group user facts
    under the same legal provision in the serialised prompt output).
    """
    scored: list[tuple[float, Chunk, Chunk]] = []

    for legal in legal_chunks:
        for user in user_chunks:
            # Score from the user chunk's perspective: what fraction of the
            # user's vocabulary appears in the legal text?  We also check the
            # reverse (legal → user) and take the max so short legal snippets
            # with low token count don't get penalised.
            score = max(
                _keyword_overlap(user.text, legal.text),
                _keyword_overlap(legal.text, user.text),
            )
            if score >= _MIN_OVERLAP:
                scored.append((score, legal, user))

    if scored:
        # Sort descending by score so the most relevant mappings appear first.
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            FactToLawMapping(
                user_chunk_id   = user.chunk_id,
                user_chunk_text = user.text[:400],
                legal_chunk_id  = legal.chunk_id,
                article_id      = legal.article_id or "",
                legal_text      = legal.text,
                source_type     = legal.source_type,
                dimension       = dimension,
            )
            for _, legal, user in scored
        ]

    # Fallback: no overlap found — include all legal chunks without a specific
    # user-chunk anchor.  The agent still gets the legal context.
    return [
        FactToLawMapping(
            user_chunk_id   = "",
            user_chunk_text = "",
            legal_chunk_id  = legal.chunk_id,
            article_id      = legal.article_id or "",
            legal_text      = legal.text,
            source_type     = legal.source_type,
            dimension       = dimension,
        )
        for legal in legal_chunks
    ]
