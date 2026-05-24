"""
debug_run.py
────────────
Pretty-print the SQLite session log produced by run.py.

Usage:
    python debug_run.py                        # reads logs/prism_run_001.db
    python debug_run.py logs/my_session.db     # specific file
    python debug_run.py --signals              # errors/signals only (fast)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SESSION_ID = "prism_run_001"

def _db_path() -> Path:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        return Path(sys.argv[1])
    return Path(f"logs/{SESSION_ID}.db")

def _flag(name: str) -> bool:
    return name in sys.argv

def _short(text: str, n: int = 120) -> str:
    text = str(text).replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text

def _load(db: Path) -> "sqlite3.Connection":
    import sqlite3
    if not db.exists():
        print(f"[error] No log file at {db}")
        print("        Run `python run.py` first — logging is now enabled.")
        sys.exit(1)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn

def main() -> None:
    import sqlite3

    db   = _db_path()
    conn = _load(db)
    print(f"\n{'━'*70}")
    print(f"  PRISM SESSION LOG  ·  {db}")
    print(f"{'━'*70}")

    # ── Pipeline events ───────────────────────────────────────────────────────
    if not _flag("--signals"):
        rows = conn.execute("SELECT * FROM pipeline_events ORDER BY id").fetchall()
        print(f"\n{'─'*70}")
        print(f"  PIPELINE EVENTS  ({len(rows)})")
        print(f"{'─'*70}")
        for r in rows:
            meta = json.loads(r["metadata"] or "{}")
            dur  = f"  [{r['duration_ms']}ms]" if r["duration_ms"] else ""
            agent = f" · {r['agent']}" if r["agent"] else ""
            print(f"  {r['timestamp'][:19]}  {r['event_type']:<30}{agent}{dur}")
            if meta:
                for k, v in meta.items():
                    print(f"    {k}: {_short(str(v), 80)}")

    # ── Signals (errors & retries) ────────────────────────────────────────────
    rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
    errors = [r for r in rows if r["signal_type"] in ("MEMORY_WRITE_ERROR", "LLM_PARSE_ERROR")]
    other  = [r for r in rows if r["signal_type"] not in ("MEMORY_WRITE_ERROR", "LLM_PARSE_ERROR")]

    if errors:
        print(f"\n{'─'*70}")
        print(f"  ⚠  ERRORS  ({len(errors)})")
        print(f"{'─'*70}")
        for r in errors:
            payload = json.loads(r["payload"] or "{}")
            dim = f" [{r['dimension']}]" if r["dimension"] else ""
            print(f"\n  {r['signal_type']}{dim}  ·  agent={r['agent']}")
            print(f"  resolution : {r['resolution']}")
            for k, v in payload.items():
                print(f"  {k:<20}: {_short(str(v))}")
    else:
        print("\n  ✓  No errors recorded.")

    if other and not _flag("--signals"):
        print(f"\n{'─'*70}")
        print(f"  SIGNALS  ({len(other)})")
        print(f"{'─'*70}")
        for r in other:
            payload = json.loads(r["payload"] or "{}")
            dim = f" [{r['dimension']}]" if r["dimension"] else ""
            print(f"\n  {r['signal_type']}{dim}  ·  agent={r['agent']}  ·  retry={r['retry_count']}")
            print(f"  resolution : {r['resolution']}")
            for k, v in payload.items():
                print(f"  {k:<20}: {_short(str(v))}")

    # ── Reasoning entries ─────────────────────────────────────────────────────
    if not _flag("--signals"):
        rows = conn.execute("SELECT * FROM reasoning_entries ORDER BY id").fetchall()
        print(f"\n{'─'*70}")
        print(f"  REASONING ENTRIES  ({len(rows)})")
        print(f"{'─'*70}")
        for r in rows:
            ok     = "✓" if r["parse_succeeded"] else "✗"
            dim    = f" [{r['dimension']}]" if r["dimension"] else ""
            conf   = f"  conf={r['confidence']}" if r["confidence"] else ""
            claims = f"  claims={r['claims_count']}" if r["claims_count"] else ""
            dur    = f"  {r['duration_ms']}ms" if r["duration_ms"] else ""
            print(f"\n  {ok} agent={r['agent']}{dim}{conf}{claims}{dur}")
            print(f"  prompt  : {_short(r['prompt_sent'])}")
            print(f"  response: {_short(r['llm_response'])}")
            if not r["parse_succeeded"]:
                print(f"  *** PARSE FAILED ***")

    # ── State changes ─────────────────────────────────────────────────────────
    if not _flag("--signals"):
        rows = conn.execute("SELECT * FROM state_changes ORDER BY id").fetchall()
        print(f"\n{'─'*70}")
        print(f"  STATE CHANGES  ({len(rows)})")
        print(f"{'─'*70}")
        for r in rows:
            ok  = "✓" if r["write_validated"] else "✗"
            errs = json.loads(r["validation_errors"] or "[]")
            print(f"\n  {ok} section={r['section']}  agent={r['agent']}")
            if errs:
                for e in errs:
                    print(f"  ✗ {e}")
            else:
                new = json.loads(r["new_state"] or "{}")
                if isinstance(new, dict):
                    for k, v in list(new.items())[:4]:
                        print(f"  {k:<22}: {_short(str(v), 60)}")

    print(f"\n{'━'*70}\n")
    conn.close()

if __name__ == "__main__":
    main()
