"""
core/log_store.py
──────────────────
SQLite backend for the Prism logging layer.

Responsibilities:
  - Open one connection per session; close it on session end.
  - Create the four log tables on first use (CREATE TABLE IF NOT EXISTS).
  - Provide one write method per table, each wrapped in try/except.
  - Write failures go to a plaintext fallback log — never raised to callers.
  - Provide parameterized read methods that return raw row dicts.
  - Use WAL mode for better concurrent read performance.

This file has no knowledge of agents, orchestrator, or pipeline logic.
All it knows is the schema and how to persist/retrieve it.
Nothing outside core/ should import this directly — all access goes through
logger.py (PrismLogger), which is the single public interface.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.log_schema import PipelineEvent, ReasoningEntry, SignalEntry, StateChangeEntry

# Python logger used when the fallback log path is not configured (e.g. :memory: mode).
_fallback_logger = logging.getLogger("prism.fallback")


class LogStore:
    """SQLite persistence layer for Prism log entries.

    Opens a single connection in initialize() and closes it in close().
    All write methods silently fall back to the plaintext error log on failure
    so that logging never interrupts the pipeline.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file, or ":memory:" for testing.
    fallback_log_path : str | None
        Path for the plaintext fallback error log.  When None (the default in
        :memory: mode), errors are written to Python's logging system instead.
    """

    def __init__(
        self,
        db_path: str,
        fallback_log_path: Optional[str] = None,
    ) -> None:
        self._db_path = db_path
        self._fallback_log_path = fallback_log_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── Setup / teardown ──────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Open the database connection and create tables if they don't exist.

        Creates the parent directory for the DB file if it doesn't exist.
        Enables WAL (Write-Ahead Logging) mode for better concurrent read
        performance — important because report export reads while the pipeline
        may still be writing.

        Stale WAL recovery
        ~~~~~~~~~~~~~~~~~~
        If a previous run was interrupted mid-write it can leave behind
        ``<db>.db-wal`` and ``<db>.db-shm`` lock files that cause
        ``sqlite3.OperationalError: disk I/O error`` on the next open.
        We detect that situation and delete the orphaned sidecar files
        before retrying — safe because the lock files are owned by a process
        that is no longer running.
        """
        if self._db_path != ":memory:":
            # Ensure the /logs directory exists before opening the file.
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False allows the connection to be used from the
        # thread that will call close() (which may differ from the thread that
        # called initialize() in async contexts).
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_tables()
        except sqlite3.OperationalError as exc:
            # Stale WAL/SHM files from an interrupted run can cause
            # "disk I/O error".  Remove them and retry exactly once.
            if self._db_path != ":memory:" and "disk I/O" in str(exc):
                for suffix in (".db-wal", ".db-shm", "-wal", "-shm"):
                    stale = Path(self._db_path + suffix)
                    if stale.exists():
                        stale.unlink()
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                # Retry after removing stale sidecar files.
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._create_tables()
            else:
                raise

    def _create_tables(self) -> None:
        """Create all four log tables using IF NOT EXISTS (idempotent)."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pipeline_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                agent       TEXT,
                metadata    TEXT,
                duration_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS reasoning_entries (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT    NOT NULL,
                timestamp         TEXT    NOT NULL,
                agent             TEXT    NOT NULL,
                dimension         TEXT,
                prompt_sent       TEXT    NOT NULL,
                llm_response      TEXT    NOT NULL,
                parsed_output     TEXT,
                parse_succeeded   INTEGER NOT NULL,
                confidence        TEXT,
                claims_count      INTEGER,
                weak_claims_count INTEGER,
                chunk_ids_used    TEXT,
                duration_ms       INTEGER
            );

            CREATE TABLE IF NOT EXISTS state_changes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT    NOT NULL,
                timestamp         TEXT    NOT NULL,
                section           TEXT    NOT NULL,
                agent             TEXT    NOT NULL,
                previous_state    TEXT,
                new_state         TEXT    NOT NULL,
                write_validated   INTEGER NOT NULL,
                validation_errors TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                agent       TEXT NOT NULL,
                dimension   TEXT,
                payload     TEXT NOT NULL,
                resolution  TEXT NOT NULL,
                retry_count INTEGER,
                resolved_at TEXT
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Fallback error log ────────────────────────────────────────────────────

    def _fallback(self, context: str, exc: Exception) -> None:
        """Write a write-failure record to the plaintext fallback log.

        Never raises — if the fallback file itself can't be written, falls
        through to Python's logging system as a last resort.  This ensures
        the pipeline is never interrupted by a logging failure.
        """
        ts = datetime.now(timezone.utc).isoformat()
        message = f"{ts} | WRITE_ERROR | {context} | {type(exc).__name__}: {exc}\n"

        if self._fallback_log_path:
            try:
                with open(self._fallback_log_path, "a", encoding="utf-8") as f:
                    f.write(message)
                return
            except OSError:
                pass  # fall through to Python logger if file write fails

        _fallback_logger.error(message.rstrip())

    # ── Write methods ─────────────────────────────────────────────────────────

    def write_pipeline_event(self, entry: PipelineEvent) -> None:
        """Insert one pipeline lifecycle event row."""
        try:
            self._conn.execute(
                "INSERT INTO pipeline_events "
                "(session_id, timestamp, event_type, agent, metadata, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.session_id,
                    entry.timestamp,
                    entry.event_type,
                    entry.agent,
                    json.dumps(entry.metadata),
                    entry.duration_ms,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            self._fallback("write_pipeline_event", exc)

    def write_reasoning_entry(self, entry: ReasoningEntry) -> None:
        """Insert one reasoning trace row."""
        try:
            self._conn.execute(
                "INSERT INTO reasoning_entries "
                "(session_id, timestamp, agent, dimension, prompt_sent, llm_response, "
                "parsed_output, parse_succeeded, confidence, claims_count, "
                "weak_claims_count, chunk_ids_used, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.session_id,
                    entry.timestamp,
                    entry.agent,
                    entry.dimension,
                    entry.prompt_sent,
                    entry.llm_response,
                    # parsed_output is None when parse failed; store as SQL NULL.
                    json.dumps(entry.parsed_output) if entry.parsed_output is not None else None,
                    int(entry.parse_succeeded),         # SQLite stores bool as 0/1
                    entry.confidence,
                    entry.claims_count,
                    entry.weak_claims_count,
                    json.dumps(entry.chunk_ids_used),   # always a list, even if empty
                    entry.duration_ms,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            self._fallback("write_reasoning_entry", exc)

    def write_state_change(self, entry: StateChangeEntry) -> None:
        """Insert one state-change row (written on both success and failure)."""
        try:
            self._conn.execute(
                "INSERT INTO state_changes "
                "(session_id, timestamp, section, agent, previous_state, new_state, "
                "write_validated, validation_errors) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.session_id,
                    entry.timestamp,
                    entry.section,
                    entry.agent,
                    # previous_state is None on first write to a section.
                    json.dumps(entry.previous_state) if entry.previous_state is not None else None,
                    json.dumps(entry.new_state),
                    int(entry.write_validated),
                    # Store empty list as NULL to save space; read back as [].
                    json.dumps(entry.validation_errors) if entry.validation_errors else None,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            self._fallback("write_state_change", exc)

    def write_signal(self, entry: SignalEntry) -> Optional[int]:
        """Insert one signal row and return the auto-increment row ID.

        The row ID is returned so the orchestrator can call resolve_signal()
        later to close the signal's lifecycle with a resolved_at timestamp.
        Returns None if the write fails (error goes to fallback log).
        """
        try:
            cursor = self._conn.execute(
                "INSERT INTO signals "
                "(session_id, timestamp, signal_type, agent, dimension, payload, "
                "resolution, retry_count, resolved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.session_id,
                    entry.timestamp,
                    entry.signal_type,
                    entry.agent,
                    entry.dimension,
                    json.dumps(entry.payload),
                    entry.resolution,
                    entry.retry_count,
                    entry.resolved_at,   # None until resolve_signal() is called
                ),
            )
            self._conn.commit()
            return cursor.lastrowid
        except Exception as exc:
            self._fallback("write_signal", exc)
            return None

    def resolve_signal(self, signal_id: int, resolved_at: str) -> None:
        """Set the resolved_at timestamp on a previously written signal row."""
        try:
            self._conn.execute(
                "UPDATE signals SET resolved_at = ? WHERE id = ?",
                (resolved_at, signal_id),
            )
            self._conn.commit()
        except Exception as exc:
            self._fallback(f"resolve_signal(id={signal_id})", exc)

    # ── Read methods ──────────────────────────────────────────────────────────
    # All read methods return list[dict] (raw row dicts).  Conversion to typed
    # dataclasses is the responsibility of PrismLogger, which owns the mapping.

    def get_pipeline_events(self, session_id: str) -> list[dict]:
        """Return all pipeline events for a session in insertion order."""
        cursor = self._conn.execute(
            "SELECT * FROM pipeline_events WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_reasoning_entries(
        self,
        session_id: str,
        agent: Optional[str] = None,
        dimension: Optional[str] = None,
    ) -> list[dict]:
        """Return reasoning entries filtered by session_id, and optionally agent/dimension."""
        query = "SELECT * FROM reasoning_entries WHERE session_id = ?"
        params: list = [session_id]
        if agent is not None:
            query += " AND agent = ?"
            params.append(agent)
        if dimension is not None:
            query += " AND dimension = ?"
            params.append(dimension)
        query += " ORDER BY id"
        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_state_history(
        self,
        session_id: str,
        section: Optional[str] = None,
    ) -> list[dict]:
        """Return state-change rows filtered by session_id and optionally section."""
        query = "SELECT * FROM state_changes WHERE session_id = ?"
        params: list = [session_id]
        if section is not None:
            query += " AND section = ?"
            params.append(section)
        query += " ORDER BY id"
        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_signals(
        self,
        session_id: str,
        signal_type: Optional[str] = None,
    ) -> list[dict]:
        """Return signal rows filtered by session_id and optionally signal_type."""
        query = "SELECT * FROM signals WHERE session_id = ?"
        params: list = [session_id]
        if signal_type is not None:
            query += " AND signal_type = ?"
            params.append(signal_type)
        query += " ORDER BY id"
        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_errors(self, session_id: str) -> list[dict]:
        """Return signal rows whose signal_type indicates an error.

        Only MEMORY_WRITE_ERROR and LLM_PARSE_ERROR are considered errors.
        Other signal types (RETRIEVAL_SIGNAL, LOOP_SIGNAL, etc.) are normal
        pipeline signals and are not returned here.
        """
        cursor = self._conn.execute(
            "SELECT * FROM signals "
            "WHERE session_id = ? "
            "AND signal_type IN ('MEMORY_WRITE_ERROR', 'LLM_PARSE_ERROR') "
            "ORDER BY id",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_sessions(self) -> list[str]:
        """Return all distinct session_ids that have at least one pipeline event."""
        cursor = self._conn.execute(
            "SELECT DISTINCT session_id FROM pipeline_events ORDER BY session_id"
        )
        return [row[0] for row in cursor.fetchall()]
