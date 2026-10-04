"""SQLite-backed storage for human review records: fields the automated
pipeline couldn't resolve, plus any correction a human reviewer makes."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/review.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id TEXT NOT NULL,
    field TEXT NOT NULL,
    model_value TEXT,
    model_confidence REAL,
    fallback_value TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | accepted | edited | rejected
    final_value TEXT,
    created_at TEXT NOT NULL,
    reviewed_at TEXT
);
"""


@contextmanager
def get_connection():
    """A context manager so every caller automatically commits and closes,
    even if an exception occurs mid-operation — avoids leaked connections
    or half-written transactions, a real risk once multiple scripts touch
    the same database file."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(SCHEMA)


def add_review_item(receipt_id: str, field: str, model_value: str | None,
                     model_confidence: float | None, fallback_value: str | None) -> int:
    """Insert a new item needing human review. Returns the new row's id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO review_items
               (receipt_id, field, model_value, model_confidence, fallback_value, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (receipt_id, field, model_value, model_confidence, fallback_value,
             datetime.now(timezone.utc).isoformat()),
        )
        return cursor.lastrowid


def get_pending_items() -> list[dict]:
    """All items awaiting human review, oldest first."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM review_items WHERE status = 'pending' ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]


def resolve_item(item_id: int, status: str, final_value: str) -> None:
    """Record a human's decision: accepted (kept model/fallback value as-is),
    edited (human provided a different value), or rejected (no usable value)."""
    if status not in {"accepted", "edited", "rejected"}:
        raise ValueError(f"Invalid status: {status!r}")
    with get_connection() as conn:
        conn.execute(
            "UPDATE review_items SET status = ?, final_value = ?, reviewed_at = ? WHERE id = ?",
            (status, final_value, datetime.now(timezone.utc).isoformat(), item_id),
        )