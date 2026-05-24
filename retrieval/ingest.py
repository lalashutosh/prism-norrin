"""
retrieval/ingest.py
───────────────────
Ingestion layer for Prism's legal corpus.

Creates and persists hierarchical **PageIndex** trees from structured legal
documents.  The PageIndex is the primary data structure for the legal corpus
retrieval path (legislation, official_guidance).  Each node represents a
natural document section boundary; parent nodes hold LLM-generated summaries
of their children so the traversal can navigate without loading full text;
leaf nodes hold the complete section text.

══════════════════════════════════════════════════════════════════════════════
PUBLIC API
══════════════════════════════════════════════════════════════════════════════
  build_index_from_json(nodes, source_type)              → PageIndexNode
  annotate_summaries(root, llm_fn)                       → None (mutates)
  save_index(root, path)                                 → None
  load_index(path)                                       → PageIndexNode
  ingest_legal_json(json_path, corpus_dir, *, ...)       → PageIndexNode

══════════════════════════════════════════════════════════════════════════════
STUBS
══════════════════════════════════════════════════════════════════════════════
  DEFAULT_SUMMARISE_FN — no-op (first 300 chars + ellipsis).
  Replace at call-site with an LLM-backed SummariseFn for production:

      import anthropic

      client = anthropic.Anthropic()

      def my_llm_summarise(text: str) -> str:
          msg = client.messages.create(
              model="claude-haiku-4-5-20251001",
              max_tokens=256,
              messages=[{
                  "role": "user",
                  "content": (
                      "Summarise this legal text in 1-2 sentences, "
                      "keeping all article references:\\n\\n" + text[:4000]
                  ),
              }],
          )
          return msg.content[0].text

      from retrieval.ingest import ingest_legal_json
      ingest_legal_json(json_path, corpus_dir, llm_fn=my_llm_summarise)

══════════════════════════════════════════════════════════════════════════════
CALLABLE CONTRACT FOR llm_fn / SummariseFn
══════════════════════════════════════════════════════════════════════════════
  signature : (text: str) -> str
  Called once per non-leaf parent node during annotate_summaries().
  Input is the concatenation of children summaries (≤ 4 000 chars typical).
  Must return a non-empty summary string.  The stub truncates the input.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ── Type aliases ─────────────────────────────────────────────────────────────

# A function that takes a text string and returns a concise summary.
SummariseFn = Callable[[str], str]


# ── Default stub ─────────────────────────────────────────────────────────────

def _default_summarise(text: str) -> str:
    """No-op summarise: return the first 300 characters of *text*.

    Suitable for offline testing.  Replace with an LLM-backed function (see
    module docstring) before building a production corpus index.
    """
    if len(text) <= 300:
        return text
    return text[:300].rstrip() + "…"


DEFAULT_SUMMARISE_FN: SummariseFn = _default_summarise


# ── Core data structure ───────────────────────────────────────────────────────

@dataclass
class PageIndexNode:
    """A single node in a hierarchical PageIndex tree.

    Leaf nodes (``children`` is empty) carry the full section body in ``text``.
    Parent nodes carry an LLM-generated ``summary`` of their children so the
    tree-traversal retriever can navigate without loading every leaf.

    Fields
    ──────
    node_id         Stable unique identifier.  Sourced from the parsed corpus
                    JSON where available; generated (via _generate_node_id)
                    for synthetic container nodes.
    title           Section heading text (e.g. "2.1. Prohibitions listed in
                    Article 5 AI Act").
    level           Depth in the tree: 0 = document root, 1 = H1, 2 = H2,
                    3 = H3.
    text            Full section body text (leaf nodes only; empty string for
                    container nodes that only group children).
    summary         Short summary used for tree-traversal scoring.  Populated
                    bottom-up by annotate_summaries(); empty until then.
    children        Ordered child nodes.
    document_source Original document basename (e.g.
                    "EU_AI_Act_Guidelines_Prohibited_Practices.pdf").
    source_type     "legislation" or "official_guidance" for corpus nodes.
    article_id      Article/annex/section reference if detectable from the
                    heading (e.g. "Article 5", "Annex III", "2.1").
    """

    node_id:         str
    title:           str
    level:           int
    text:            str
    summary:         str
    children:        list["PageIndexNode"] = field(default_factory=list)
    document_source: str                   = ""
    source_type:     str                   = "official_guidance"
    article_id:      Optional[str]         = None


# ── Tree construction ─────────────────────────────────────────────────────────

def build_index_from_json(
    nodes: list[dict[str, Any]],
    source_type: str = "official_guidance",
) -> PageIndexNode:
    """Reconstruct a PageIndexNode tree from the flat JSON produced by
    ``parse_eu_ai_guidelines.py`` (compile_atomic_nodes output).

    Input format (per element)
    ──────────────────────────
    {
      "node_id":        "GL_H2_...",
      "node_label":     "Guideline_Rule",
      "hierarchy":      ["1. TITLE", "1.1. Sub-section"],  # len 2 or 3
      "atomic_content": "Full text …",
      "document_source": "EU_AI_Act_…pdf"
    }

    Strategy
    ────────
    Build a *trie* keyed by ``tuple(hierarchy_path)``.  For each atomic node,
    walk from root to parent depth, creating synthetic container nodes on
    demand, then insert the leaf at the full path.

    Returns a synthetic root (level=0) whose children are the H1 sections.
    """
    if not nodes:
        return PageIndexNode(
            node_id="root",
            title="Document",
            level=0,
            text="",
            summary="",
        )

    # Use the first node's document_source as the tree-level default.
    doc_source = nodes[0].get("document_source", "")

    root = PageIndexNode(
        node_id="root",
        title="Document",
        level=0,
        text="",
        summary="",
        document_source=doc_source,
        source_type=source_type,
    )

    # Maps tuple(hierarchy_path) → the PageIndexNode at that depth.
    # The empty tuple maps to root.
    node_by_path: dict[tuple[str, ...], PageIndexNode] = {(): root}

    for raw in nodes:
        hierarchy: list[str] = raw.get("hierarchy", [])
        content:   str       = raw.get("atomic_content", "")
        node_id:   str       = raw.get("node_id", "") or _generate_node_id(hierarchy)
        raw_src:   str       = raw.get("document_source", doc_source)

        if not hierarchy:
            continue

        # ── Create any missing ancestor container nodes ────────────────────
        # E.g. for hierarchy ["H1", "H2", "H3"] we ensure ("H1",) and
        # ("H1", "H2") exist before we insert ("H1", "H2", "H3").
        for depth in range(1, len(hierarchy)):
            anc_path = tuple(hierarchy[:depth])
            if anc_path in node_by_path:
                continue  # already created

            parent_path = tuple(hierarchy[: depth - 1])
            parent_node = node_by_path[parent_path]
            title       = hierarchy[depth - 1]

            ancestor = PageIndexNode(
                node_id         = f"ctr_{_generate_node_id(list(anc_path))}",
                title           = title,
                level           = depth,
                text            = "",      # containers have no body text
                summary         = "",      # filled by annotate_summaries()
                document_source = raw_src,
                source_type     = source_type,
                article_id      = _extract_article_id(list(anc_path)),
            )
            parent_node.children.append(ancestor)
            node_by_path[anc_path] = ancestor

        # ── Insert the leaf node at the full path ─────────────────────────
        full_path   = tuple(hierarchy)
        parent_path = tuple(hierarchy[:-1])
        parent_node = node_by_path.get(parent_path, root)

        # Guard against duplicate paths (e.g. repeated section headings in the PDF).
        if full_path in node_by_path:
            # Append content to the existing node rather than shadowing it.
            existing = node_by_path[full_path]
            if content and content not in existing.text:
                existing.text = (existing.text + "\n\n" + content).strip()
            continue

        leaf = PageIndexNode(
            node_id         = node_id,
            title           = hierarchy[-1],
            level           = len(hierarchy),
            text            = content,
            summary         = "",          # filled by annotate_summaries()
            document_source = raw_src,
            source_type     = source_type,
            article_id      = _extract_article_id(hierarchy),
        )
        parent_node.children.append(leaf)
        node_by_path[full_path] = leaf

    return root


# ── Summary annotation ────────────────────────────────────────────────────────

def annotate_summaries(
    root: PageIndexNode,
    llm_fn: SummariseFn = DEFAULT_SUMMARISE_FN,
) -> None:
    """Compute summaries for every node via a bottom-up (post-order) walk.

    Leaf nodes
    ~~~~~~~~~~
    ``summary = llm_fn(text)`` — so the traversal can score leaves by their
    summary rather than loading all leaf text into memory at once.  Skipped if
    ``summary`` is already non-empty (idempotent on re-run).

    Parent nodes
    ~~~~~~~~~~~~
    ``summary = llm_fn(concatenation of children's summaries)`` — the
    concatenation is capped at 4 000 characters to stay within LLM context.

    Mutates the tree in-place; returns None.
    """
    def _walk(node: PageIndexNode) -> None:
        for child in node.children:
            _walk(child)                     # children first (post-order)

        if node.summary:
            return                           # already annotated — skip

        if not node.children:
            # Leaf: summarise own body text (or fall back to the title).
            source = node.text or node.title
            node.summary = llm_fn(source) if source else node.title
            return

        # Parent: summarise from children's summaries (or their titles).
        child_texts: list[str] = []
        for child in node.children:
            snippet = child.summary or child.title
            if snippet:
                child_texts.append(snippet)

        if child_texts:
            combined = "\n\n".join(child_texts)
            # Cap at 4 000 chars to avoid over-long LLM prompts.
            node.summary = llm_fn(combined[:4000])

    _walk(root)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index(root: PageIndexNode, path: Path) -> None:
    """Serialise the PageIndexNode tree to *path* as a JSON file.

    The envelope carries version and top-level metadata so ``load_index``
    can validate the format without inspecting the tree.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope: dict[str, Any] = {
        "version":          "1.0",
        "source_type":      root.source_type,
        "document_source":  root.document_source,
        "tree":             _node_to_dict(root),
    }
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index(path: Path) -> PageIndexNode:
    """Deserialise a PageIndexNode tree from a file written by ``save_index``.

    Raises
    ──────
    FileNotFoundError  if the file does not exist.
    ValueError         if the JSON envelope is missing the expected ``"tree"``
                       key (unrecognised format).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PageIndex file not found: {path}")

    envelope: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "tree" not in envelope:
        raise ValueError(
            f"Unrecognised PageIndex format in {path} "
            "(expected top-level 'tree' key)."
        )
    return _dict_to_node(envelope["tree"])


# ── Convenience pipeline ──────────────────────────────────────────────────────

def ingest_legal_json(
    json_path: Path,
    corpus_dir: Path,
    *,
    source_type: str = "official_guidance",
    llm_fn: SummariseFn = DEFAULT_SUMMARISE_FN,
) -> PageIndexNode:
    """Full ingestion pipeline: JSON file → annotated PageIndex → saved to disk.

    Steps
    ─────
    1. Load the flat JSON array from *json_path*.
    2. Build the hierarchical PageIndexNode tree via build_index_from_json().
    3. Annotate all nodes bottom-up with summaries via annotate_summaries().
    4. Save the serialised tree to *corpus_dir* / <stem>.pageindex.json.
    5. Return the root PageIndexNode (immediately usable for retrieval).

    The saved file name is derived from the input JSON stem:
        ``eu_ai_act_prohibited_practices_atomic_nodes.json``
        → ``corpus/eu_ai_act_prohibited_practices_atomic_nodes.pageindex.json``

    Re-running overwrites the existing index file.

    Parameters
    ──────────
    json_path   Path to the flat JSON array produced by parse_eu_ai_guidelines.py.
    corpus_dir  Directory where the serialised .pageindex.json is written.
    source_type "official_guidance" or "legislation" (propagated to all nodes).
    llm_fn      Summarisation function.  DEFAULT_SUMMARISE_FN is a stub; pass
                a real LLM-backed function for production indices.
    """
    json_path  = Path(json_path)
    corpus_dir = Path(corpus_dir)

    raw_nodes: list[dict[str, Any]] = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    root = build_index_from_json(raw_nodes, source_type=source_type)
    annotate_summaries(root, llm_fn=llm_fn)

    out_path = corpus_dir / f"{json_path.stem}.pageindex.json"
    save_index(root, out_path)

    return root


# ── Private helpers ───────────────────────────────────────────────────────────

# Article / annex / section reference patterns (priority order).
_ARTICLE_RE = re.compile(
    r"Article\s+\d+[a-z]?(?:\(\d+\))?(?:\([a-z]+\))?",
    re.IGNORECASE,
)
_ANNEX_RE = re.compile(
    r"Annex\s+[IVXLivxl]+(?:\s+(?:para\w*|paragraph)\s*\d+)?",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)")


def _extract_article_id(hierarchy: list[str]) -> Optional[str]:
    """Extract the most specific legal reference from a hierarchy list.

    Checks the deepest title first, then walks upward toward the root.
    Priority: Article match > Annex match > section number (e.g. "2.1").

    Returns None if no reference is found.
    """
    for title in reversed(hierarchy):
        m = _ARTICLE_RE.search(title)
        if m:
            return m.group(0)
        m = _ANNEX_RE.search(title)
        if m:
            return m.group(0)

    # Section number fallback (e.g. "2.1." from "2.1. Prohibited Practices").
    if hierarchy:
        m = _SECTION_RE.match(hierarchy[-1].strip())
        if m:
            return m.group(1).rstrip(".")

    return None


def _generate_node_id(hierarchy: list[str]) -> str:
    """Derive a stable, unique node_id from a hierarchy path list.

    Used only for synthetic container nodes that have no ``node_id`` in the
    source JSON.  The BLAKE2s digest guarantees uniqueness across similar titles.
    """
    raw    = " / ".join(hierarchy)
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=4).hexdigest()
    slug   = re.sub(r"[^a-z0-9]+", "_", raw.lower())[:60].strip("_")
    return f"{slug}_{digest}"


def _node_to_dict(node: PageIndexNode) -> dict[str, Any]:
    """Recursively serialise a PageIndexNode to a JSON-serialisable dict."""
    return {
        "node_id":          node.node_id,
        "title":            node.title,
        "level":            node.level,
        "text":             node.text,
        "summary":          node.summary,
        "document_source":  node.document_source,
        "source_type":      node.source_type,
        "article_id":       node.article_id,
        "children":         [_node_to_dict(c) for c in node.children],
    }


def _dict_to_node(d: dict[str, Any]) -> PageIndexNode:
    """Recursively deserialise a plain dict into a PageIndexNode."""
    return PageIndexNode(
        node_id         = d.get("node_id", ""),
        title           = d.get("title", ""),
        level           = int(d.get("level", 0)),
        text            = d.get("text", ""),
        summary         = d.get("summary", ""),
        document_source = d.get("document_source", ""),
        source_type     = d.get("source_type", "official_guidance"),
        article_id      = d.get("article_id"),
        children        = [_dict_to_node(c) for c in d.get("children", [])],
    )
