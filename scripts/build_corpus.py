"""
scripts/build_corpus.py
───────────────────────
Standalone corpus ingestion script.

Scans retrieval/*_atomic_nodes.json, ingests any files that don't already
have a corresponding .pageindex.json in retrieval/corpus/, and writes the
new index files.  Already-indexed files are silently skipped (idempotent).

Usage:
    python scripts/build_corpus.py [--fast] [--force]

Options:
    --fast   Use the truncation summariser instead of LLM calls.
             Faster; acceptable quality because legal text is already
             keyword-rich.  The retriever still finds relevant nodes via
             keyword overlap on the full article text.
    --force  Re-index ALL files, even those that already have .pageindex.json.

Environment variables (same as run.py):
    LLM_BASE_URL   http://localhost:8000/v1
    LLM_MODEL      Qwen/Qwen3.6-27B
    LLM_API_KEY    none
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RETRIEVAL_DIR = Path("retrieval")
CORPUS_DIR    = Path("retrieval/corpus")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "Qwen/Qwen3.6-27B")
LLM_API_KEY  = os.environ.get("LLM_API_KEY",  "none")


def _make_llm_summarise(llm_client, model: str, total: int):
    """Return a summarise fn that calls the LLM with a progress counter."""
    counter = [0]

    def summarise(text: str) -> str:
        import openai
        counter[0] += 1
        preview = text[:55].replace("\n", " ")
        print(f"  node {counter[0]:>3}/{total} — {preview!r:.55s} …", flush=True)
        for attempt in range(5):
            try:
                resp = llm_client.messages.create(
                    model      = model,
                    max_tokens = 256,
                    system     = (
                        "You are a legal document summariser. "
                        "Produce a single concise sentence (≤ 40 words) summarising "
                        "the key legal requirement or topic of the passage below."
                    ),
                    messages   = [{"role": "user", "content": text[:4000]}],
                )
                return resp.content[0].text.strip()
            except openai.RateLimitError:
                if attempt == 4:
                    raise
                wait = 6.0 * (2 ** attempt)
                print(f"  Rate limit — retrying in {wait:.0f}s …")
                time.sleep(wait)
        return text[:300]

    return summarise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Prism legal corpus index.")
    parser.add_argument("--fast",  action="store_true",
                        help="Use truncation summariser (no LLM calls).")
    parser.add_argument("--force", action="store_true",
                        help="Re-index even already-indexed files.")
    args = parser.parse_args()

    from retrieval.ingest import ingest_legal_json, DEFAULT_SUMMARISE_FN

    json_files = sorted(RETRIEVAL_DIR.glob("*_atomic_nodes.json"))
    if not json_files:
        print("No *_atomic_nodes.json files found in retrieval/")
        return

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # Build LLM client only if needed.
    llm_client = None
    if not args.fast:
        from core.llm_client import make_llm_client
        print(f"Connecting to LLM at {LLM_BASE_URL} (model: {LLM_MODEL}) …")
        llm_client = make_llm_client(
            "openai_compat",
            model             = LLM_MODEL,
            api_key           = LLM_API_KEY,
            base_url          = LLM_BASE_URL,
            max_output_tokens = 256,
            disable_thinking  = True,
        )

    for json_path in json_files:
        out_path = CORPUS_DIR / f"{json_path.stem}.pageindex.json"
        if out_path.exists() and not args.force:
            print(f"[skip]  {json_path.name} → {out_path.name} already exists")
            continue

        print(f"\n[index] {json_path.name} …")
        raw_nodes   = json.loads(json_path.read_text(encoding="utf-8"))
        total_nodes = len(raw_nodes) + len(raw_nodes) // 5

        # Determine source_type from filename.
        stem_lower  = json_path.stem.lower()
        source_type = (
            "official_guidance"
            if any(kw in stem_lower for kw in ("prohibited", "guideline", "guidance"))
            else "legislation"
        )
        print(f"  {len(raw_nodes)} atomic nodes  source_type={source_type!r}")

        if args.fast or llm_client is None:
            llm_fn = DEFAULT_SUMMARISE_FN
            print("  Using truncation summariser (--fast mode)")
        else:
            llm_fn = _make_llm_summarise(llm_client, LLM_MODEL, total_nodes)

        root = ingest_legal_json(
            json_path,
            CORPUS_DIR,
            source_type = source_type,
            llm_fn      = llm_fn,
        )
        print(f"  → Saved {out_path.name}  (root: {root.node_id!r})")

    n_total = len(list(CORPUS_DIR.glob("*.pageindex.json")))
    print(f"\nCorpus ready: {n_total} index file(s) in {CORPUS_DIR}/")


if __name__ == "__main__":
    main()
