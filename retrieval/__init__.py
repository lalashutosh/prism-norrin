"""retrieval — Prism dual-stream retrieval layer.

Public entry points
───────────────────
  retrieval.retriever.retrieve(query, filters) → list[Chunk]
      Single function consumed by the Orchestrator.

  retrieval.ingest.ingest_legal_json(...)
      CLI / offline pipeline step: JSON → PageIndex tree → corpus dir.

Nothing in this package imports from agents, memory, or logger.
"""
