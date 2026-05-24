"""
retrieval/upload_ingest.py
──────────────────────────
User-upload ingestion pipeline for Prism.

Processes user-supplied PDFs, text files, and images into searchable chunks,
then persists the chunks, dense-vector embeddings, and session metadata to a
session-scoped local directory for hybrid BM25 + semantic retrieval.

══════════════════════════════════════════════════════════════════════════════
PUBLIC API
══════════════════════════════════════════════════════════════════════════════
  extract_image_context(image_bytes, *, filename, vision_fn) → str
      Stub for Vision-LLM / OCR image understanding.  See docstring for the
      Anthropic claude-haiku injection pattern.

  embed_texts(texts, *, embed_fn) → list[list[float]]
      Stub wrapper for dense-vector embedding.  See docstring for injection.

  extract_file_text(path, *, vision_fn) → list[SegmentDict]
      Route a file to the appropriate extractor; returns a list of content
      segments, each: {"text": str, "page": int, "is_image": bool}.

  chunk_text(text, chunk_size, overlap) → list[tuple[str, int, int]]
      Split text into overlapping word-token windows.
      Returns (chunk_text, char_start, char_end) triples.

  ingest_files(file_paths, session_id, *, embed_fn, vision_fn, ...) → IngestionResult
      Full pipeline: extract → chunk → embed → persist.  Single entry point
      for the application layer.

  load_session(session_id) → tuple[list[Chunk], list[list[float]]]
      Deserialise a persisted session into (chunks, vectors) ready for
      retrieval.  Returns ([], []) when the session does not exist.

  clear_session(session_id) → None
      Delete the session directory and all its data.

══════════════════════════════════════════════════════════════════════════════
STUBS (injectable)
══════════════════════════════════════════════════════════════════════════════
  DEFAULT_VISION_FN
      Returns a placeholder description string.  Replace via vision_fn=… in
      ingest_files() or extract_file_text():

          import anthropic, base64
          client = anthropic.Anthropic()

          def my_vision(image_bytes: bytes) -> str:
              b64 = base64.standard_b64encode(image_bytes).decode()
              msg = client.messages.create(
                  model="claude-haiku-4-5-20251001",
                  max_tokens=512,
                  messages=[{
                      "role": "user",
                      "content": [
                          {"type": "image", "source": {
                              "type": "base64",
                              "media_type": "image/jpeg",
                              "data": b64,
                          }},
                          {"type": "text", "text":
                              "Describe this image precisely, noting any text, "
                              "charts, tables, or diagrams relevant to AI compliance."},
                      ],
                  }],
              )
              return msg.content[0].text

          ingest_files(paths, session_id, vision_fn=my_vision)

  DEFAULT_EMBED_FN
      Returns zero vectors.  BM25 dominates RRF while this stub is active.
      Replace via embed_fn=… in ingest_files():

          import openai
          client = openai.OpenAI()

          def my_embed(texts: list[str]) -> list[list[float]]:
              resp = client.embeddings.create(
                  model="text-embedding-3-small", input=texts
              )
              return [r.embedding for r in resp.data]

          ingest_files(paths, session_id, embed_fn=my_embed)

══════════════════════════════════════════════════════════════════════════════
SESSION STORAGE LAYOUT  (retrieval/user_dbs/{session_id}/)
══════════════════════════════════════════════════════════════════════════════
  chunks.json   — JSON array of ChunkRecord dicts (text + provenance)
  vectors.json  — JSON array of float arrays, one per chunk (same order)
  meta.json     — session-level metadata (file count, doc names, timestamps)
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Only cross-package import allowed: core.types shared types.
# Never import agents, memory, logger, or retriever.
from core.types import Chunk


# ── Type aliases ──────────────────────────────────────────────────────────────

# A function that accepts raw image bytes and returns a descriptive string.
VisionFn = Callable[[bytes], str]

# A function that accepts a list of strings and returns a list of float vectors.
EmbedFn = Callable[[list[str]], list[list[float]]]

# Internal segment dict returned by file extractors.
SegmentDict = dict[str, Any]   # {"text": str, "page": int, "is_image": bool, ...}


# ── Configuration ─────────────────────────────────────────────────────────────

# Root directory for all session databases.
USER_DB_DIR: Path = Path(__file__).parent / "user_dbs"

# Default chunking parameters (word-token based).
DEFAULT_CHUNK_SIZE: int = 500   # words per chunk
DEFAULT_OVERLAP:    int = 50    # word overlap between consecutive chunks

# Supported file extensions.
_PDF_EXT:    str           = ".pdf"
_IMAGE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff",
})


# ── Default stubs ─────────────────────────────────────────────────────────────

def _default_vision_fn(image_bytes: bytes) -> str:
    """No-op vision stub: returns a size-annotated placeholder.

    Replace by passing ``vision_fn=my_vision`` to ingest_files().
    See the module docstring for the Anthropic claude-haiku injection pattern.
    """
    size_kb = max(1, len(image_bytes) // 1024)
    return (
        f"[Image ({size_kb} KB) — vision analysis stub. "
        "Inject a real VisionFn via ingest_files(vision_fn=...) for content extraction.]"
    )


def _default_embed_fn(texts: list[str]) -> list[list[float]]:
    """No-op embedding stub: returns a zero vector per text.

    Replace by passing ``embed_fn=my_embed`` to ingest_files().
    See the module docstring for the OpenAI text-embedding-3-small pattern.
    BM25 rank dominates the RRF merge while this stub is active.
    """
    return [[0.0] for _ in texts]


DEFAULT_VISION_FN: VisionFn = _default_vision_fn
DEFAULT_EMBED_FN:  EmbedFn  = _default_embed_fn


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """Internal record for one chunk of an uploaded document.

    Distinct from core.types.Chunk to carry provenance fields (char offsets,
    page numbers, image flag) without polluting the shared type contract.
    Converted to core.types.Chunk at retrieval time via _record_to_chunk().

    Fields
    ──────
    chunk_id        Stable, session-scoped identifier.
    text            The chunk body (may be vision-LLM description for images).
    doc_name        Original filename (basename, e.g. "policy.pdf").
    page_numbers    0-indexed page number(s) covered by this chunk.
    char_start      Character offset (inclusive) in the source segment text.
    char_end        Character offset (exclusive) in the source segment text.
    is_image_chunk  True when the text came from vision/OCR rather than direct
                    text extraction.
    chunk_index     Ordinal position across all chunks in the session.
    """
    chunk_id:       str
    text:           str
    doc_name:       str
    page_numbers:   list[int]
    char_start:     int
    char_end:       int
    is_image_chunk: bool = False
    chunk_index:    int  = 0


@dataclass
class IngestionResult:
    """Summary returned by ingest_files() once all files are processed.

    Attributes
    ──────────
    session_id   The session identifier that was used.
    file_count   Number of files that were successfully processed.
    chunk_count  Total number of chunks produced across all files.
    doc_names    List of original filenames (basenames), in processing order.
    session_dir  Absolute path to the session database directory.
    """
    session_id:  str
    file_count:  int
    chunk_count: int
    doc_names:   list[str]
    session_dir: str


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC STUBS
# ══════════════════════════════════════════════════════════════════════════════

def extract_image_context(
    image_bytes: bytes,
    *,
    filename: str = "",
    vision_fn: VisionFn = DEFAULT_VISION_FN,
) -> str:
    """Convert raw image bytes to a textual description via Vision-LLM or OCR.

    This is the central injection point for image understanding.  By default
    it calls DEFAULT_VISION_FN, which returns a placeholder string.

    Parameters
    ──────────
    image_bytes  Raw bytes of the image (PNG, JPEG, etc.).
    filename     Optional source filename prepended to the description for
                 provenance in the final retrieval results.
    vision_fn    Pluggable vision function.  Signature: (bytes) → str.
                 See the module docstring for the claude-haiku injection pattern.

    Returns
    ───────
    str  Human-readable description of the image content.
    """
    description = vision_fn(image_bytes)
    if filename:
        return f"[{filename}] {description}"
    return description


def embed_texts(
    texts: list[str],
    *,
    embed_fn: EmbedFn = DEFAULT_EMBED_FN,
) -> list[list[float]]:
    """Generate dense-vector embeddings for a list of text strings.

    This is the injection point for embedding models.  By default it calls
    DEFAULT_EMBED_FN, which returns zero vectors.

    Parameters
    ──────────
    texts     Strings to embed.  May be empty (returns []).
    embed_fn  Pluggable embedding function.  Signature: (list[str]) → list[list[float]].
              See the module docstring for the OpenAI text-embedding-3-small pattern.

    Returns
    ───────
    list[list[float]]  One float vector per input string.
    """
    if not texts:
        return []
    return embed_fn(texts)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_file_text(
    path: Path,
    *,
    vision_fn: VisionFn = DEFAULT_VISION_FN,
) -> list[SegmentDict]:
    """Extract content segments from a file, routing by extension.

    Returns a list of SegmentDicts.  Each dict has at minimum:
        "text"     : str   — the text content (may be vision description)
        "page"     : int   — 0-indexed page / section number
        "is_image" : bool  — True if text came from a vision stub

    Routing
    ───────
    .pdf            →  _extract_pdf_content (text + embedded-image detection)
    image extension →  _extract_image_file  (vision stub on whole file)
    anything else   →  _extract_txt_file    (UTF-8 plain-text read)
    """
    path   = Path(path)
    suffix = path.suffix.lower()

    if suffix == _PDF_EXT:
        return _extract_pdf_content(path, vision_fn=vision_fn)
    if suffix in _IMAGE_EXTS:
        return _extract_image_file(path, vision_fn=vision_fn)
    return _extract_txt_file(path)


def _extract_pdf_content(
    path: Path,
    *,
    vision_fn: VisionFn = DEFAULT_VISION_FN,
) -> list[SegmentDict]:
    """Extract text (and optionally vision descriptions) from a PDF.

    For each page:
    - Extracts text when present.
    - Detects image-heavy pages (< 20 words AND pdfplumber.page.images is
      non-empty) and calls _render_page_for_vision() → vision stub.
    - Pages with both substantial text AND embedded images: extract text
      and set the ``has_embedded_images`` flag in the segment dict so
      downstream processors can note the mixed content.

    Falls back to reading the file as UTF-8 plain text if pdfplumber is not
    installed or raises an exception.
    """
    segments: list[SegmentDict] = []

    # ── pdfplumber path ─────────────────────────────────────────────────────
    try:
        import pdfplumber  # noqa: PLC0415

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text  = (page.extract_text() or "").strip()
                has_images = bool(page.images)
                word_count = len(page_text.split())

                # An image-dominated page has few words and embedded images.
                if has_images and word_count < 20:
                    vision_text = _render_page_for_vision(
                        page, path, page_num, vision_fn
                    )
                    segments.append({
                        "text":     vision_text,
                        "page":     page_num,
                        "is_image": True,
                    })
                    # Include whatever sparse text was found (e.g. captions).
                    if page_text:
                        segments.append({
                            "text":     page_text,
                            "page":     page_num,
                            "is_image": False,
                        })
                elif page_text:
                    seg: SegmentDict = {
                        "text":     page_text,
                        "page":     page_num,
                        "is_image": False,
                    }
                    if has_images:
                        # Note for the application layer; not used in retrieval.
                        seg["has_embedded_images"] = True
                    segments.append(seg)

        return segments

    except ImportError:
        pass   # fall through to plain-text fallback
    except Exception:
        pass

    # ── Plain-text fallback ──────────────────────────────────────────────────
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            segments.append({"text": content, "page": 0, "is_image": False})
    except Exception:
        pass

    return segments


def _render_page_for_vision(
    page: Any,
    path: Path,
    page_num: int,
    vision_fn: VisionFn,
) -> str:
    """Render a pdfplumber Page to PNG bytes and call the vision stub.

    Requires Pillow and a PDF renderer (poppler or ghostscript) to be installed.
    Falls back to a descriptive placeholder when rendering is unavailable.
    """
    try:
        # page.to_image() returns a pdfplumber PageImage backed by PIL.
        pil_image = page.to_image(resolution=150).original
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        return extract_image_context(
            image_bytes,
            filename=f"{path.name}:p{page_num + 1}",
            vision_fn=vision_fn,
        )
    except Exception:
        # Pillow or the PDF renderer is not available.
        return (
            f"[{path.name} page {page_num + 1}: image-heavy page. "
            "Install Pillow and a PDF renderer (poppler/ghostscript) for "
            "automatic vision extraction, or inject a VisionFn.]"
        )


def _extract_image_file(
    path: Path,
    *,
    vision_fn: VisionFn = DEFAULT_VISION_FN,
) -> list[SegmentDict]:
    """Read a standalone image file and call the vision stub."""
    try:
        image_bytes = path.read_bytes()
        description = extract_image_context(
            image_bytes, filename=path.name, vision_fn=vision_fn
        )
        return [{"text": description, "page": 0, "is_image": True}]
    except Exception:
        return [{
            "text":     f"[Could not read image file: {path.name}]",
            "page":     0,
            "is_image": True,
        }]


def _extract_txt_file(path: Path) -> list[SegmentDict]:
    """Read a plain-text file as a single content segment."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            return [{"text": content, "page": 0, "is_image": False}]
        return []
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int    = DEFAULT_OVERLAP,
) -> list[tuple[str, int, int]]:
    """Split *text* into overlapping word-token windows.

    Algorithm
    ─────────
    1. Locate all non-whitespace tokens in *text* via regex, recording their
       character start/end positions.
    2. Slide a window of *chunk_size* tokens across the token list, advancing
       by ``chunk_size - overlap`` tokens each step.
    3. Each window is materialised by slicing the original string from the
       first token's start to the last token's end — preserving all internal
       whitespace including newlines.

    Parameters
    ──────────
    text        Source text to chunk.
    chunk_size  Window size in word-tokens (default 500).
    overlap     Token overlap between consecutive windows (default 50).
                Must be < chunk_size; clamped to 0 if not.

    Returns
    ───────
    list[tuple[str, int, int]]  — (chunk_text, char_start, char_end) triples.
    char_start is inclusive; char_end is exclusive (standard Python slice).
    Returns [] for whitespace-only input.
    """
    if not text.strip():
        return []

    # Build a list of (char_start, char_end) for every non-whitespace token.
    boundaries: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in re.finditer(r"\S+", text)
    ]
    if not boundaries:
        return []

    total_tokens = len(boundaries)
    # Guard: overlap must be strictly less than chunk_size to make progress.
    safe_overlap = max(0, min(overlap, chunk_size - 1))
    step         = chunk_size - safe_overlap

    chunks: list[tuple[str, int, int]] = []
    word_start = 0

    while word_start < total_tokens:
        word_end   = min(word_start + chunk_size, total_tokens)
        char_start = boundaries[word_start][0]
        char_end   = boundaries[word_end - 1][1]   # inclusive end of last token
        chunk_body = text[char_start:char_end]

        if chunk_body.strip():
            chunks.append((chunk_body, char_start, char_end))

        if word_end >= total_tokens:
            break
        word_start += step

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def ingest_files(
    file_paths: list[str | Path],
    session_id: str,
    *,
    embed_fn:   EmbedFn  = DEFAULT_EMBED_FN,
    vision_fn:  VisionFn = DEFAULT_VISION_FN,
    chunk_size: int      = DEFAULT_CHUNK_SIZE,
    overlap:    int      = DEFAULT_OVERLAP,
) -> IngestionResult:
    """Ingest a list of files into a session-scoped retrieval database.

    Pipeline
    ────────
    1. **Extract** raw content from each file via extract_file_text().
    2. **Chunk** each content segment with word-token overlap windows.
    3. **Embed** all chunk texts in a single batch via embed_fn().
    4. **Persist** chunks, vectors, and metadata to the session directory.

    If a file cannot be read (e.g. missing path), it is skipped silently and
    does not count toward file_count in the result.

    Parameters
    ──────────
    file_paths  Paths to the files to ingest (PDF, image, or text).
    session_id  Caller-defined session identifier.  Used as the database
                directory name and as a component of chunk IDs.
    embed_fn    Dense-vector embedding function (default: zero-vector stub).
    vision_fn   Vision-LLM / OCR function for image content (default: stub).
    chunk_size  Tokens per chunk window (default 500).
    overlap     Token overlap between windows (default 50).

    Returns
    ───────
    IngestionResult with counts and the path to the session directory.
    """
    all_records: list[ChunkRecord] = []
    doc_names:   list[str]         = []
    global_idx:  int               = 0

    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue

        doc_name = path.name
        doc_names.append(doc_name)

        # ── Extract content segments ─────────────────────────────────────────
        segments = extract_file_text(path, vision_fn=vision_fn)

        # ── Chunk each segment ───────────────────────────────────────────────
        for segment in segments:
            seg_text:     str  = segment.get("text", "")
            seg_page:     int  = segment.get("page", 0)
            seg_is_image: bool = segment.get("is_image", False)

            raw_chunks = chunk_text(seg_text, chunk_size=chunk_size, overlap=overlap)

            for chunk_body, char_start, char_end in raw_chunks:
                if not chunk_body.strip():
                    continue
                cid = _make_chunk_id(session_id, doc_name, global_idx)
                all_records.append(ChunkRecord(
                    chunk_id       = cid,
                    text           = chunk_body,
                    doc_name       = doc_name,
                    page_numbers   = [seg_page],
                    char_start     = char_start,
                    char_end       = char_end,
                    is_image_chunk = seg_is_image,
                    chunk_index    = global_idx,
                ))
                global_idx += 1

    # ── Embed all chunks in one batch ────────────────────────────────────────
    vectors: list[list[float]]
    if all_records:
        texts   = [r.text for r in all_records]
        vectors = embed_fn(texts)
        # Defensive: pad with zero vectors if embed_fn returns fewer rows.
        while len(vectors) < len(all_records):
            vectors.append([0.0])
        vectors = vectors[: len(all_records)]   # truncate if over-produced
    else:
        vectors = []

    # ── Persist to session directory ─────────────────────────────────────────
    meta = {
        "session_id":  session_id,
        "file_count":  len(doc_names),
        "chunk_count": len(all_records),
        "doc_names":   doc_names,
    }
    _save_session(session_id, all_records, vectors, meta)

    return IngestionResult(
        session_id  = session_id,
        file_count  = len(doc_names),
        chunk_count = len(all_records),
        doc_names   = doc_names,
        session_dir = str(_session_dir(session_id)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_session(
    session_id: str,
) -> tuple[list[Chunk], list[list[float]]]:
    """Load a persisted session's chunks and embedding vectors from disk.

    Returns
    ───────
    (list[Chunk], list[list[float]])
        Chunks are ``core.types.Chunk`` objects with ``source_type="uploaded_doc"``
        and ``article_id=None``.  Vectors are in the same order as chunks.
        Returns ([], []) when the session does not exist or the files are corrupt.
    """
    sdir         = _session_dir(session_id)
    chunks_file  = sdir / "chunks.json"
    vectors_file = sdir / "vectors.json"

    if not chunks_file.exists() or not vectors_file.exists():
        return [], []

    try:
        records_raw: list[dict] = json.loads(
            chunks_file.read_text(encoding="utf-8")
        )
        vectors: list[list[float]] = json.loads(
            vectors_file.read_text(encoding="utf-8")
        )

        records = [ChunkRecord(**d) for d in records_raw]
        chunks  = [_record_to_chunk(r) for r in records]

        # Guard: if vector count mismatches, truncate to the shorter.
        n = min(len(chunks), len(vectors))
        return chunks[:n], vectors[:n]

    except Exception:  # noqa: BLE001
        return [], []


def clear_session(session_id: str) -> None:
    """Delete the session directory and all its persisted data.

    A no-op when the session directory does not exist.
    """
    sdir = _session_dir(session_id)
    if sdir.exists():
        shutil.rmtree(sdir)


# ── Private persistence helpers ───────────────────────────────────────────────

def _session_dir(session_id: str) -> Path:
    """Return the session-specific database directory path."""
    # Sanitise session_id to prevent directory traversal.
    safe_id = re.sub(r"[^\w\-]", "_", session_id)[:128]
    return USER_DB_DIR / safe_id


def _save_session(
    session_id: str,
    records:    list[ChunkRecord],
    vectors:    list[list[float]],
    meta:       dict,
) -> None:
    """Persist chunks, vectors, and metadata to the session directory.

    Files written
    ─────────────
    chunks.json   JSON array of ChunkRecord dicts.
    vectors.json  JSON array of float arrays (one per chunk).
    meta.json     Session-level metadata dict (pretty-printed).

    Existing files are silently overwritten (idempotent on re-ingest).
    """
    sdir = _session_dir(session_id)
    sdir.mkdir(parents=True, exist_ok=True)

    (sdir / "chunks.json").write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False),
        encoding="utf-8",
    )
    (sdir / "vectors.json").write_text(
        json.dumps(vectors),
        encoding="utf-8",
    )
    (sdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _record_to_chunk(record: ChunkRecord) -> Chunk:
    """Convert an internal ChunkRecord to a core.types.Chunk for retrieval.

    Constraints enforced here (per spec):
    - source_type  = "uploaded_doc"  (always)
    - article_id   = None            (always)
    - provenance fields go into metadata{}
    """
    return Chunk(
        chunk_id    = record.chunk_id,
        text        = record.text,
        source_type = "uploaded_doc",
        article_id  = None,
        metadata    = {
            "doc_name":       record.doc_name,
            "page_numbers":   record.page_numbers,
            "char_start":     record.char_start,
            "char_end":       record.char_end,
            "is_image_chunk": record.is_image_chunk,
            "chunk_index":    record.chunk_index,
        },
    )


def _make_chunk_id(session_id: str, doc_name: str, chunk_index: int) -> str:
    """Generate a stable, collision-resistant chunk ID.

    Format: ``ud_<blake2s-hex>`` where the digest covers the full
    (session_id, doc_name, chunk_index) triple.

    The ``ud_`` prefix signals uploaded-doc provenance at a glance.
    """
    raw    = f"{session_id}\x00{doc_name}\x00{chunk_index}"
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=8).hexdigest()
    return f"ud_{digest}"
