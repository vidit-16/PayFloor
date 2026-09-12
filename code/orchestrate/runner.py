"""Row execution: concurrent, checkpointed, and impossible to lose work with.

Two failures kill hackathon runs. One: a crash at row 90 of 110 throws away an
hour of paid model calls. Two: one malformed row raises and the whole run dies,
so you ship nothing. This module fixes both — every completed row is appended to
a JSONL checkpoint immediately, and a row that raises is logged and replaced
with a conservative fallback rather than propagating.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

Row = Mapping[str, Any]
ProcessFn = Callable[[Row], dict[str, Any]]


@dataclass
class RowFailure:
    key: str
    error: str
    trace: str


@dataclass
class RunResult:
    rows: list[dict[str, Any]]
    failures: list[RowFailure] = field(default_factory=list)
    resumed: int = 0
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failures


class Checkpoint:
    """Append-only JSONL of completed rows, keyed so a rerun resumes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._done: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._done[str(record.get("__key"))] = record

    def has(self, key: str) -> bool:
        return key in self._done

    def get(self, key: str) -> dict[str, Any] | None:
        record = self._done.get(key)
        if record is None:
            return None
        return {k: v for k, v in record.items() if k != "__key"}

    def put(self, key: str, row: dict[str, Any]) -> None:
        with self._lock:
            self._done[key] = {"__key": key, **row}
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"__key": key, **row}, default=str) + "\n")

    def clear(self) -> None:
        self._done.clear()
        if self.path.exists():
            self.path.unlink()

    def __len__(self) -> int:
        return len(self._done)


class Runner:
    """Maps `process` over input rows with bounded concurrency."""

    def __init__(
        self,
        *,
        key_column: str,
        checkpoint_path: str | Path,
        max_concurrency: int = 6,
        fallback: Callable[[Row, Exception], dict[str, Any]] | None = None,
        verbose: bool = True,
    ) -> None:
        self.key_column = key_column
        self.checkpoint = Checkpoint(checkpoint_path)
        self.max_concurrency = max(1, max_concurrency)
        self.fallback = fallback
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    def run(self, rows: Sequence[Row], process: ProcessFn) -> RunResult:
        started = time.perf_counter()
        results: dict[str, dict[str, Any]] = {}
        failures: list[RowFailure] = []
        pending: list[tuple[str, Row]] = []
        resumed = 0

        for row in rows:
            key = str(row.get(self.key_column, "")).strip()
            cached = self.checkpoint.get(key)
            if cached is not None:
                results[key] = cached
                resumed += 1
            else:
                pending.append((key, row))

        if resumed:
            self._log(f"[runner] resumed {resumed} row(s) from checkpoint")
        self._log(f"[runner] processing {len(pending)} row(s) at concurrency {self.max_concurrency}")

        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {pool.submit(process, row): (key, row) for key, row in pending}
            for future in as_completed(futures):
                key, row = futures[future]
                try:
                    output = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad row must not end the run
                    failures.append(
                        RowFailure(key=key, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
                    )
                    self._log(f"[runner] FAILED {key}: {type(exc).__name__}: {exc}")
                    output = self.fallback(row, exc) if self.fallback else {}
                results[key] = output
                self.checkpoint.put(key, output)
                completed += 1
                if self.verbose and completed % 10 == 0:
                    self._log(f"[runner] {completed}/{len(pending)}")

        ordered = [
            {self.key_column: str(row.get(self.key_column, "")).strip(),
             **results.get(str(row.get(self.key_column, "")).strip(), {})}
            for row in rows
        ]
        elapsed = time.perf_counter() - started
        self._log(
            f"[runner] done in {elapsed:.1f}s - {len(ordered)} rows, {len(failures)} failure(s)"
        )
        return RunResult(rows=ordered, failures=failures, resumed=resumed, elapsed=elapsed)
