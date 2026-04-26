"""Usage tracking and cost estimation backed by SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Unique per process launch – groups rows from a single app session.
_SESSION_ID: str = uuid.uuid4().hex[:12]

# Default cost rates (cents).  Override via config.yaml → usage.cost_rates
_DEFAULT_RATES: dict[str, float] = {
    "llm_per_1k_tokens_cents": 2.0,
    "tts_per_1k_chars_cents": 30.0,
    "video_per_minute_cents": 0.0,  # local GPU – no API cost
}


class UsageTracker:
    """Lightweight SQLite usage logger.  Thread-safe via short-lived connections."""

    # ------------------------------------------------------------------
    # Performance Profiling & Predictive Logging (AIOps)
    # ------------------------------------------------------------------

    def log_latency(self, phase: str, duration_s: float, threshold_s: float = 5.0, metadata: dict | None = None):
        """Log latency for a generation phase. Warn if above threshold."""
        self._insert(
            f"latency_{phase}",
            duration_s=duration_s,
            metadata=metadata,
        )
        if duration_s > threshold_s:
            logger.warning(f"Latency Warning: {phase} phase took {duration_s:.2f}s (>{threshold_s}s)")
            self._insert(
                f"latency_warning_{phase}",
                duration_s=duration_s,
                metadata={"warning": True, **(metadata or {})},
            )

    def __init__(self, db_path: str = "output/usage.db", config: Optional[dict[str, Any]] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._rates = dict(_DEFAULT_RATES)
        if config:
            overrides = config.get("usage", {}).get("cost_rates", {})
            self._rates.update({k: v for k, v in overrides.items() if k in self._rates})
        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            # Enable WAL mode idempotently
            try:
                wal_status = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
                logger.info(f"SQLite WAL mode enabled: {wal_status[0]}")
            except Exception as e:
                logger.warning(f"Failed to enable WAL mode: {e}")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    session_id      TEXT    NOT NULL,
                    operation       TEXT    NOT NULL,
                    provider        TEXT,
                    tokens_used     INTEGER,
                    characters_generated INTEGER,
                    video_duration_seconds REAL,
                    estimated_cost_cents REAL,
                    metadata        TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_op ON usage(operation)")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _insert(self, operation: str, *, provider: str = "", tokens: int = 0,
                characters: int = 0, duration_s: float = 0.0,
                cost_cents: float = 0.0, metadata: dict | None = None):
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO usage
                       (timestamp, session_id, operation, provider,
                        tokens_used, characters_generated, video_duration_seconds,
                        estimated_cost_cents, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        _SESSION_ID,
                        operation,
                        provider,
                        tokens or None,
                        characters or None,
                        duration_s or None,
                        round(cost_cents, 2),
                        json.dumps(metadata) if metadata else None,
                    ),
                )
            logger.info(f"Usage logged | op: {operation} | provider: {provider} | tokens: {tokens} | chars: {characters} | duration: {duration_s} | cost: {cost_cents}")
        except Exception as e:
            logger.error(f"Usage logging failed | op: {operation} | error: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_llm_call(self, provider: str, model: str,
                     prompt_tokens: int = 0, completion_tokens: int = 0,
                     total_tokens: int = 0):
        total = total_tokens or (prompt_tokens + completion_tokens)
        cost = total / 1000.0 * self._rates["llm_per_1k_tokens_cents"]
        self._insert(
            "llm_call", provider=provider, tokens=total,
            cost_cents=cost,
            metadata={"model": model, "prompt_tokens": prompt_tokens,
                       "completion_tokens": completion_tokens},
        )

    def log_tts_call(self, provider: str, characters: int, voice_id: str = ""):
        cost = characters / 1000.0 * self._rates["tts_per_1k_chars_cents"]
        self._insert(
            "tts_call", provider=provider, characters=characters,
            cost_cents=cost,
            metadata={"voice_id": voice_id},
        )

    def log_video_generation(self, duration_s: float, resolution: str = "",
                             fps: int = 0, pipeline_type: str = ""):
        cost = (duration_s / 60.0) * self._rates["video_per_minute_cents"]
        self._insert(
            "video_generation", duration_s=duration_s,
            cost_cents=cost,
            metadata={"resolution": resolution, "fps": fps,
                       "pipeline_type": pipeline_type},
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_summary(self, since: Optional[str] = None) -> dict[str, Any]:
        """Return aggregate totals, optionally filtered by ISO timestamp."""
        clause = ""
        params: list[str] = []
        if since:
            clause = " WHERE timestamp >= ?"
            params.append(since)

        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT operation,
                           COUNT(*)                          AS count,
                           COALESCE(SUM(tokens_used), 0)     AS total_tokens,
                           COALESCE(SUM(characters_generated), 0) AS total_chars,
                           COALESCE(SUM(video_duration_seconds), 0) AS total_video_s,
                           COALESCE(SUM(estimated_cost_cents), 0)  AS total_cost_cents
                    FROM usage{clause}
                    GROUP BY operation""",
                params,
            ).fetchall()

        summary: dict[str, Any] = {"operations": {}, "total_cost_cents": 0.0}
        for row in rows:
            op = row["operation"]
            summary["operations"][op] = {
                "count": row["count"],
                "total_tokens": row["total_tokens"],
                "total_chars": row["total_chars"],
                "total_video_s": round(row["total_video_s"], 1),
                "total_cost_cents": round(row["total_cost_cents"], 2),
            }
            summary["total_cost_cents"] += row["total_cost_cents"]
        summary["total_cost_cents"] = round(summary["total_cost_cents"], 2)
        return summary

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent usage rows."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM usage ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_tracker: Optional[UsageTracker] = None


def get_tracker(config: Optional[dict[str, Any]] = None) -> UsageTracker:
    """Return (or create) the module-level UsageTracker singleton."""
    global _tracker  # noqa: PLW0603
    if _tracker is None:
        db_path = "output/usage.db"
        if config:
            db_path = config.get("usage", {}).get("db_path", db_path)
            output_dir = config.get("app", {}).get("output_dir", "")
            if output_dir and db_path == "output/usage.db":
                db_path = str(Path(output_dir) / "usage.db")
        _tracker = UsageTracker(db_path=db_path, config=config)
    return _tracker
