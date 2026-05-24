"""
retrieval/retriever.py
──────────────────────
Public retrieval interface for Prism's dual-stream RAG layer.

Only the **Orchestrator** calls this module; agents never import it.

══════════════════════════════════════════════════════════════════════════════
PUBLIC INTERFACE
══════════════════════════════════════════════════════════════════════════════
  retrieve(query, filters, session_id)  → list[Chunk]
      Sole function consumed by the Orchestrator.  session_id is optional;
      when supplied the user-document path loads from the persisted session
      database produced by upload_ingest.ingest_files().

  register_document(doc_id, chunks)     ← legacy in-memory user-doc path
  clear_document_store()                ← reset _DOC_STORE for test isolation
  preload_legal_index(root)             ← inject a pre-built PageIndexNode
  set_embed_fn(fn)                      ← swap the embedding stub
  invalidate_session_cache(session_id)  ← evict one or all session caches
  _reset_corpus_cache()                 ← test helper; resets lazy-load state

══════════════════════════════════════════════════════════════════════════════
ROUTING LOGIC
══════════════════════════════════════════════════════════════════════════════
  filters["source_types"] : list[str]  (preferred key, set by agents)
  filters["source_type"]  : str        (scalar alternative)

  "legislation" | "official_guidance"  →  PageIndex tree-traversal path
  "uploaded_doc"                       →  BM25 + semantic hybrid (RRF)
       if session_id is given: load chunks + pre-computed vectors from disk
       otherwise: use in-memory _DOC_STORE (legacy / test path)
  absent / empty                       →  both paths merged

══════════════════════════════════════════════════════════════════════════════
USER-DOCUMENT RETRIEVAL (session_id path)
══════════════════════════════════════════════════════════════════════════════
  When session_id is provided:
  1. Load (chunks, vectors) from the session DB via load_session() with an
     in-process LRU-style cache (_SESSION_CACHE) to avoid repeated I/O.
  2. Compute BM25 ranking over chunk texts.
  3. Compute semantic ranking using PRE-COMPUTED document vectors — only the
     query is embedded at retrieval time, not the full corpus.
  4. Merge via RRF; return top-_TOP_K_RESULTS Chunks.

  To update the orchestrator to use sessions:
      from functools import partial
      retrieve_fn = partial(retrieve, session_id=session_id)
      orchestrator = Orchestrator(retrieve_fn=retrieve_fn, ...)

══════════════════════════════════════════════════════════════════════════════
STUBS (injectable)
══════════════════════════════════════════════════════════════════════════════
  _stub_embed(texts) → list[list[float]]
      Returns zero vectors.  BM25 rank dominates RRF merge until replaced.
      Inject a real function via set_embed_fn():

          import openai
          client = openai.OpenAI()

          def real_embed(texts: list[str]) -> list[list[float]]:
              resp = client.embeddings.create(
                  model="text-embedding-3-small", input=texts
              )
              return [r.embedding for r in resp.data]

          from retrieval.retriever import set_embed_fn
          set_embed_fn(real_embed)

══════════════════════════════════════════════════════════════════════════════
MODULE-LEVEL STATE
══════════════════════════════════════════════════════════════════════════════
  _INDEX_CACHE    : lazy-loaded PageIndexNode trees, keyed by file stem.
  _DOC_STORE      : in-memory user-document chunks (legacy / test path).
  _SESSION_CACHE  : (chunks, vectors) per session_id; avoids repeated disk I/O.
  _EMBED_FN       : pluggable embedding function (starts as zero-vector stub).
  _CORPUS_LOADED  : flag preventing repeated corpus-dir scans.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

# Only allowed cross-package imports: core.types (shared types) and
# retrieval.ingest (same package).  Never import agents, memory, or logger.
from core.types import Chunk
from retrieval.ingest import PageIndexNode, load_index


# ── Configuration ─────────────────────────────────────────────────────────────

# Default corpus directory (sibling to this file).
CORPUS_DIR: Path = Path(__file__).parent / "corpus"

# Source type groupings.
_LEGAL_SOURCE_TYPES: frozenset[str] = frozenset({"legislation", "official_guidance"})
_USER_SOURCE_TYPES:  frozenset[str] = frozenset({"uploaded_doc"})

# Tree-traversal tuning.
_TOP_K_CHILDREN: int = 5   # how many child nodes to recurse into per level
_MAX_DEPTH:      int = 5   # absolute recursion ceiling (guards malformed trees)

# BM25 (Okapi BM25) parameters.
_BM25_K1: float = 1.5
_BM25_B:  float = 0.75

# Reciprocal Rank Fusion constant k (higher k → more rank smoothing).
_RRF_K: int = 60

# Final result limit returned from retrieve().
_TOP_K_RESULTS: int = 10


# ── Module-level state ────────────────────────────────────────────────────────

# Legal corpus trees, lazy-loaded from CORPUS_DIR on first access.
_INDEX_CACHE:   dict[str, PageIndexNode] = {}
_CORPUS_LOADED: bool                     = False

# User-uploaded document chunks registered per session.
# Key: caller-supplied doc_id; Value: list of Chunk objects.
_DOC_STORE: dict[str, list[Chunk]] = {}

# Session-scoped retrieval cache: session_id → (chunks, pre-computed vectors).
# Populated lazily on first retrieve() call for a given session_id.
# Evict with invalidate_session_cache() after upload_ingest.clear_session().
_SESSION_CACHE: dict[str, tuple[list[Chunk], list[list[float]]]] = {}

# Pluggable embedding function type.
EmbedFn = Callable[[list[str]], list[list[float]]]


def _stub_embed(texts: list[str]) -> list[list[float]]:
    """Stub embedding function — returns a unit zero vector per text.

    Semantic ranking produces equal scores for all documents when this stub
    is active, so BM25 rank dominates the RRF merge — a safe degraded mode.

    Replace with a real function via set_embed_fn() for production use.
    See the module docstring for an OpenAI example.
    """
    # All-zero vectors cannot be unit-normalised; cosine returns 0.0 uniformly.
    return [[0.0] for _ in texts]


_EMBED_FN: EmbedFn = _stub_embed


# ── Public configuration API ──────────────────────────────────────────────────

def set_embed_fn(fn: EmbedFn) -> None:
    """Replace the embedding stub with a real implementation.

    The function must accept a non-empty list of strings and return a list of
    float vectors, all with the same dimension.  The first element of the
    returned list corresponds to the first input string.
    """
    global _EMBED_FN
    _EMBED_FN = fn


def register_document(doc_id: str, chunks: list[Chunk]) -> None:
    """Register user-uploaded document chunks for hybrid retrieval.

    Calling this multiple times with the same *doc_id* replaces the previous
    registration — the expected pattern when a user re-uploads a document.

    Parameters
    ──────────
    doc_id  Caller-defined identifier for this document (e.g. session ID +
            filename).  Used only as a dict key internally.
    chunks  List of Chunk objects produced from the uploaded document.
            chunk_id values must be unique within the session.
    """
    _DOC_STORE[doc_id] = list(chunks)


def clear_document_store() -> None:
    """Remove all registered user-document chunks.

    Intended for test isolation and session teardown.  Does not affect the
    legal corpus cache.
    """
    _DOC_STORE.clear()


def preload_legal_index(root: PageIndexNode) -> None:
    """Inject a pre-built PageIndexNode into the corpus cache.

    Bypasses corpus-dir I/O; useful for:
    - Unit / integration tests that supply a synthetic tree.
    - Hot-loading a freshly ingested document without a process restart.

    Also sets _CORPUS_LOADED = True so the lazy corpus scan is not triggered.
    """
    global _CORPUS_LOADED
    _INDEX_CACHE[root.document_source or root.node_id] = root
    _CORPUS_LOADED = True


def invalidate_session_cache(session_id: Optional[str] = None) -> None:
    """Evict one or all sessions from the in-process retrieval cache.

    Call this after upload_ingest.clear_session() (or after re-ingesting a
    session) to ensure stale chunks and vectors are not served.

    Parameters
    ──────────
    session_id  If given, evict only that session.
                If None (default), evict all cached sessions.
    """
    if session_id is None:
        _SESSION_CACHE.clear()
    else:
        _SESSION_CACHE.pop(session_id, None)


def _reset_corpus_cache() -> None:
    """Reset the lazy-loaded corpus cache.

    **For testing only.**  Clears _INDEX_CACHE and resets _CORPUS_LOADED so
    the next retrieve() call re-scans CORPUS_DIR.
    """
    global _CORPUS_LOADED
    _INDEX_CACHE.clear()
    _CORPUS_LOADED = False


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(
    query:      str,
    filters:    dict,
    session_id: Optional[str] = None,
) -> list[Chunk]:
    """Retrieve relevant Chunks for *query* subject to *filters*.

    This is the single function consumed by the Orchestrator's retrieve_fn.

    Parameters
    ──────────
    query       Natural-language retrieval query.
    filters     Routing and dimension metadata from the agent signal.
                Recognised keys:
                  source_types : list[str] — preferred; multi-source routing.
                  source_type  : str       — scalar alternative.
                  dimension    : str       — informational; not used for routing.
                Unrecognised keys are ignored.
    session_id  Optional session identifier.  When provided and the
                user-document stream is active, chunks and pre-computed
                vectors are loaded from the persisted session DB rather than
                the in-memory _DOC_STORE.  Pass via functools.partial or a
                closure from the application layer:

                    from functools import partial
                    retrieve_fn = partial(retrieve, session_id=session_id)

    Returns
    ───────
    list[Chunk]  Up to _TOP_K_RESULTS items, deduplicated by chunk_id,
                 most relevant first.  Returns [] when no corpus/docs loaded.
    """
    requested: set[str] = _parse_source_types(filters)

    # If no source_types filter is present, search both streams.
    want_legal = not requested or bool(requested & _LEGAL_SOURCE_TYPES)
    want_user  = not requested or bool(requested & _USER_SOURCE_TYPES)

    results: list[Chunk] = []

    if want_legal:
        results.extend(_legal_retrieve(query, filters))

    if want_user:
        results.extend(_doc_retrieve(query, filters, session_id=session_id))

    # Deduplicate by chunk_id, preserving insertion order.
    seen:    set[str]    = set()
    deduped: list[Chunk] = []
    for chunk in results:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            deduped.append(chunk)

    return deduped[:_TOP_K_RESULTS]


# ══════════════════════════════════════════════════════════════════════════════
# LEGAL CORPUS RETRIEVAL — PageIndex tree traversal
# ══════════════════════════════════════════════════════════════════════════════

def _legal_retrieve(query: str, filters: dict) -> list[Chunk]:
    """Navigate all loaded PageIndex trees and return the most relevant Chunks.

    Steps
    ─────
    1. Ensure the corpus cache is populated (lazy I/O on first call).
    2. Determine which source_types are requested.  When the caller specifies
       a subset (e.g. ["legislation"] only), skip trees whose root source_type
       is not in the requested set so Article 5 official_guidance does not
       bleed into non-prohibited-practices dimension queries.
    3. For each selected corpus tree, call _tree_traverse() to collect leaves.
    4. Score every candidate against the query using keyword overlap.
    5. Return the top-_TOP_K_RESULTS leaves as Chunk objects.
    """
    _ensure_corpus_loaded()
    if not _INDEX_CACHE:
        return []

    # Parse the requested legal source types.  If the caller asked for
    # ["legislation", "official_guidance"] (or left source_types empty) we
    # search all trees.  If only ["legislation"] is requested, we skip trees
    # whose root.source_type is "official_guidance" (and vice versa).
    requested_legal: set[str] = _parse_source_types(filters) & _LEGAL_SOURCE_TYPES
    # Empty requested_legal means "no filter" — search all legal trees.

    candidates: list[tuple[float, PageIndexNode]] = []

    for root in _INDEX_CACHE.values():
        # Skip this corpus tree when the caller has an explicit source_type
        # filter that does not include this tree's source_type.
        if requested_legal and root.source_type not in requested_legal:
            continue
        leaves = _tree_traverse(query, root, depth=0)
        for leaf in leaves:
            score = _keyword_score(query, leaf.summary or leaf.text)
            candidates.append((score, leaf))

    # Sort descending by relevance score.
    candidates.sort(key=lambda t: t[0], reverse=True)

    return [_node_to_chunk(leaf) for _, leaf in candidates[:_TOP_K_RESULTS]]


def _tree_traverse(
    query: str,
    node: PageIndexNode,
    depth: int,
) -> list[PageIndexNode]:
    """Recursively navigate the PageIndex tree; collect content nodes.

    A *content node* is any node that carries text — whether a pure leaf (no
    children) or a container that has both its own body text AND children
    (e.g. a section with an intro paragraph and numbered sub-sections).

    At each level:
      1. Score every child's ``summary`` (or first 500 chars of text, or title)
         against the query.
      2. Recurse into the top-_TOP_K_CHILDREN scoring children.
      3. If the current container node (depth > 0) has its own body text,
         also include it as a result so its intro content is not lost.
      4. Pure leaf nodes (no children) with content are collected directly.

    The depth cap (_MAX_DEPTH) prevents infinite recursion on pathological trees.

    Returns
    ───────
    list[PageIndexNode]  — content nodes to convert to Chunks.
    """
    if depth > _MAX_DEPTH:
        return []

    if not node.children:
        # Pure leaf: include only if it has content.
        return [node] if (node.text or node.summary) else []

    # Score all children against the query.
    scored: list[tuple[float, PageIndexNode]] = [
        (
            _keyword_score(
                query,
                child.summary or child.text[:500] or child.title,
            ),
            child,
        )
        for child in node.children
    ]
    scored.sort(key=lambda t: t[0], reverse=True)

    # Recurse into the most relevant children only.
    results: list[PageIndexNode] = []
    for _, child in scored[:_TOP_K_CHILDREN]:
        results.extend(_tree_traverse(query, child, depth + 1))

    # A container with its own text (e.g. a section with intro + sub-sections)
    # is also a content node.  Exclude depth=0 (synthetic document root).
    if depth > 0 and node.text:
        results.append(node)

    return results


def _keyword_score(query: str, text: str) -> float:
    """Keyword overlap score: |query_terms ∩ text_terms| / |query_terms|.

    Used during tree traversal for fast directional scoring.  When node
    summaries are LLM-generated this reliably navigates to relevant subtrees
    without requiring embedding calls.

    Returns 0.0 for empty query or text.
    """
    if not query or not text:
        return 0.0
    q_terms = set(_tokenize(query))
    t_terms = set(_tokenize(text))
    if not q_terms:
        return 0.0
    return len(q_terms & t_terms) / len(q_terms)


# ══════════════════════════════════════════════════════════════════════════════
# USER DOCUMENT RETRIEVAL — BM25 + semantic hybrid with RRF
# ══════════════════════════════════════════════════════════════════════════════

def _doc_retrieve(
    query:      str,
    filters:    dict,
    session_id: Optional[str] = None,
) -> list[Chunk]:
    """Hybrid BM25 + semantic retrieval over user-document chunks.

    Routing
    ───────
    session_id provided  →  load (chunks, pre-computed vectors) from the
                            persisted session DB via _session_retrieve().
    session_id absent    →  use in-memory _DOC_STORE (legacy / test path),
                            embedding all chunk texts at query time.

    Steps (both paths)
    ──────────────────
    1. Obtain a flat list of chunks and their embedding vectors.
    2. BM25 ranking over chunk texts.
    3. Semantic ranking via cosine similarity.
    4. RRF merge; return top-_TOP_K_RESULTS Chunks.
    """
    # ── Session-DB path (persistent, pre-computed vectors) ───────────────────
    if session_id:
        return _session_retrieve(query, session_id)

    # ── In-memory path (legacy / test) ───────────────────────────────────────
    all_chunks: list[Chunk] = [
        chunk
        for chunks in _DOC_STORE.values()
        for chunk in chunks
    ]
    if not all_chunks:
        return []

    texts = [chunk.text for chunk in all_chunks]

    # BM25 ranking — returns indices sorted best-first.
    bm25 = _BM25(texts, k1=_BM25_K1, b=_BM25_B)
    bm25_ranked: list[int] = [idx for _, idx in bm25.rank(query)]

    # Semantic ranking — embeds all texts + query at retrieval time.
    sem_ranked: list[int] = _semantic_rank(query, texts)

    # Reciprocal Rank Fusion merge.
    merged: list[int] = _rrf_merge([bm25_ranked, sem_ranked], k=_RRF_K)

    return [all_chunks[i] for i in merged[:_TOP_K_RESULTS]]


def _session_retrieve(query: str, session_id: str) -> list[Chunk]:
    """Hybrid retrieval using chunks and pre-computed vectors from a session DB.

    Loads the session once and caches it in _SESSION_CACHE to avoid repeated
    disk I/O on every retrieve() call within the same process lifetime.

    Steps
    ─────
    1. Load (chunks, vectors) from cache or disk.
    2. BM25 ranking over chunk texts.
    3. Semantic ranking: embed only the query; cosine-similarity to stored vecs.
    4. RRF merge; return top-_TOP_K_RESULTS Chunks.
    """
    chunks, vectors = _load_session_cached(session_id)
    if not chunks:
        return []

    texts = [c.text for c in chunks]

    # BM25 ranking.
    bm25        = _BM25(texts, k1=_BM25_K1, b=_BM25_B)
    bm25_ranked = [idx for _, idx in bm25.rank(query)]

    # Semantic ranking using pre-computed document vectors.
    sem_ranked = _semantic_rank_with_vectors(query, vectors)

    merged = _rrf_merge([bm25_ranked, sem_ranked], k=_RRF_K)
    return [chunks[i] for i in merged[:_TOP_K_RESULTS]]


def _load_session_cached(
    session_id: str,
) -> tuple[list[Chunk], list[list[float]]]:
    """Return (chunks, vectors) for *session_id*, loading from disk if needed.

    The first successful load is stored in _SESSION_CACHE so subsequent calls
    within the same process are served from memory.

    Calls upload_ingest.load_session() via a lazy import to avoid creating a
    module-level circular dependency between retriever ↔ upload_ingest.
    """
    if session_id in _SESSION_CACHE:
        return _SESSION_CACHE[session_id]

    from retrieval.upload_ingest import load_session  # noqa: PLC0415
    result = load_session(session_id)
    _SESSION_CACHE[session_id] = result
    return result


def _semantic_rank(query: str, texts: list[str]) -> list[int]:
    """Rank *texts* by cosine similarity to *query* using _EMBED_FN.

    Embeds query and all documents in one batch call.  Used for the in-memory
    _DOC_STORE path where no pre-computed vectors are available.

    When _EMBED_FN is the stub (zero vectors), all cosine scores are 0.0 and
    the order is undefined — RRF then relies on BM25 rank for tie-breaking,
    which is the intended safe-degradation behaviour.
    """
    if not texts:
        return []

    # Embed query + all documents in one call to amortise API round-trips.
    all_vecs = _EMBED_FN([query] + texts)
    q_vec    = all_vecs[0]
    doc_vecs = all_vecs[1:]

    scores = [_cosine(q_vec, dv) for dv in doc_vecs]
    return sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)


def _semantic_rank_with_vectors(
    query:             str,
    precomputed_vecs:  list[list[float]],
) -> list[int]:
    """Rank documents by cosine similarity using pre-computed embedding vectors.

    Only the *query* is embedded at retrieval time; document vectors were
    computed once at ingest time and persisted to disk.  This avoids
    re-embedding the full corpus on every retrieve() call.

    Falls back gracefully when the stub embed returns zero vectors (all
    cosine scores are 0.0 → BM25 rank dominates RRF).

    Parameters
    ──────────
    query             Natural-language query string.
    precomputed_vecs  Float vectors, one per document chunk, in the same
                      order as the chunk list.

    Returns
    ───────
    list[int]  Indices into *precomputed_vecs* sorted by cosine score
               descending (best match first).
    """
    if not precomputed_vecs:
        return []

    q_vecs = _EMBED_FN([query])
    if not q_vecs:
        return list(range(len(precomputed_vecs)))

    q_vec  = q_vecs[0]
    scores = [_cosine(q_vec, dv) for dv in precomputed_vecs]
    return sorted(range(len(precomputed_vecs)), key=lambda i: scores[i], reverse=True)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors.

    Returns 0.0 for mismatched lengths or zero-norm vectors.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ══════════════════════════════════════════════════════════════════════════════
# BM25 — Okapi BM25 ranking
# ══════════════════════════════════════════════════════════════════════════════

class _BM25:
    """Okapi BM25 ranker over a fixed, pre-tokenised corpus.

    Parameters
    ──────────
    corpus  list of document strings to rank.
    k1      term-frequency saturation parameter (default 1.5).
    b       length-normalisation parameter (default 0.75).

    The IDF formula used is the BM25+ variant:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    which guarantees IDF ≥ 0 for all terms, avoiding negative contributions
    from high-frequency terms in small corpora.
    """

    def __init__(
        self,
        corpus: list[str],
        k1: float = 1.5,
        b:  float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b  = b
        self.N  = len(corpus)

        tokenized: list[list[str]] = [_tokenize(doc) for doc in corpus]
        self.avgdl: float = (
            sum(len(t) for t in tokenized) / max(1, self.N)
        )

        # Document frequency per term (count of docs containing that term).
        df: dict[str, int] = defaultdict(int)
        for doc_tokens in tokenized:
            for term in set(doc_tokens):
                df[term] += 1

        # Pre-compute IDF for all vocabulary terms.
        self._idf: dict[str, float] = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

        # Per-document term-frequency counters and lengths.
        self._tf: list[Counter[str]] = [Counter(t) for t in tokenized]
        self._dl: list[int]          = [len(t) for t in tokenized]

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score of document *doc_idx* against *query*."""
        tf  = self._tf[doc_idx]
        dl  = self._dl[doc_idx]
        k1  = self.k1
        b   = self.b

        sc = 0.0
        for term in _tokenize(query):
            idf_val = self._idf.get(term, 0.0)
            if idf_val == 0.0:
                continue
            tf_val = tf.get(term, 0)
            sc += idf_val * (tf_val * (k1 + 1)) / (
                tf_val + k1 * (1.0 - b + b * dl / self.avgdl)
            )
        return sc

    def rank(self, query: str) -> list[tuple[float, int]]:
        """Return (score, doc_index) pairs sorted by score descending."""
        scored = [(self.score(query, i), i) for i in range(self.N)]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored


# ══════════════════════════════════════════════════════════════════════════════
# RRF — Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════

def _rrf_merge(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion across multiple ranked document-index lists.

    Each list contains document indices sorted best-first.  RRF score for
    document d:
        rrf(d) = Σ_r  1 / (k + rank_r(d))

    Documents appearing in more ranked lists and at higher positions score
    higher.  Returns a merged list of doc indices sorted by rrf() descending.

    Parameters
    ──────────
    ranked_lists  One list per retrieval stream, each a list of doc indices
                  sorted best-first.
    k             RRF smoothing constant (default 60, following the original
                  Cormack et al. 2009 paper).
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked, start=1):
            scores[doc_idx] += 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens from *text*."""
    return _TOKEN_RE.findall(text.lower())


def _parse_source_types(filters: dict) -> set[str]:
    """Extract the set of requested source types from a filters dict.

    Handles both the list form (``source_types``) and the scalar form
    (``source_type``).  Returns an empty set when neither key is present.
    """
    raw = filters.get("source_types") or filters.get("source_type") or []
    if isinstance(raw, str):
        raw = [raw]
    return set(raw)


def _node_to_chunk(node: PageIndexNode) -> Chunk:
    """Convert a PageIndexNode leaf into the Chunk type used by agents.

    The node_id becomes chunk_id.  Extra node metadata (title, level,
    document_source) is stored in the Chunk.metadata dict rather than
    inventing new Chunk fields.
    """
    return Chunk(
        chunk_id    = node.node_id,
        text        = node.text or node.summary,
        source_type = node.source_type,
        article_id  = node.article_id,
        metadata    = {
            "title":           node.title,
            "level":           node.level,
            "document_source": node.document_source,
        },
    )


def _ensure_corpus_loaded() -> None:
    """Scan CORPUS_DIR for *.pageindex.json files and populate _INDEX_CACHE.

    This is the lazy I/O path.  Called automatically on the first legal
    retrieve.  Subsequent calls are no-ops (_CORPUS_LOADED flag).

    Corrupt or unreadable index files are silently skipped so a single bad
    file does not prevent other valid indexes from loading.
    """
    global _CORPUS_LOADED
    if _CORPUS_LOADED:
        return

    # Set the flag before I/O to prevent re-entry if load_index() throws.
    _CORPUS_LOADED = True

    if not CORPUS_DIR.is_dir():
        return

    for index_file in sorted(CORPUS_DIR.glob("*.pageindex.json")):
        try:
            root = load_index(index_file)
            _INDEX_CACHE[index_file.stem] = root
        except Exception:  # noqa: BLE001
            # Partial corpus is better than no corpus.
            pass
