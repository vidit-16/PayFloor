"""Content-addressed cache for model calls.

Why this exists: `temperature=0` does not make an LLM pipeline reproducible. The
only way to get a byte-identical rerun is to not make the call twice. Every model
call is keyed on a SHA-256 of (model, system, payload), so a rerun after a crash,
a code change in the policy layer, or a re-score costs nothing and returns
exactly what the first run returned.

SQLite rather than a directory of JSON files: one file to copy around, atomic
writes, and no filesystem pressure from tens of thousands of tiny files.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key         TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL,
    model       TEXT NOT NULL,
    value       TEXT NOT NULL,
    usage       TEXT,
    created_at  REAL NOT NULL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS idx_calls_namespace ON calls(namespace);
"""


def content_key(namespace: str, model: str, payload: Any) -> str:
    """Stable hash of a call. `payload` is JSON-serialized with sorted keys so
    that dict ordering can never silently produce a cache miss."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{namespace}\x00{model}\x00{blob}".encode())
    return digest.hexdigest()


class CallCache:
    """Thread-safe persistent cache. Safe to share across the runner's workers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[Any, dict[str, Any]] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, usage FROM calls WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0]), json.loads(row[1] or "{}")

    def put(
        self,
        key: str,
        namespace: str,
        model: str,
        value: Any,
        usage: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls (key, namespace, model, value, usage) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    namespace,
                    model,
                    json.dumps(value, ensure_ascii=False, default=str),
                    json.dumps(usage or {}),
                ),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        return {"entries": total, "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> CallCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
