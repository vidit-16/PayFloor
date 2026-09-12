"""Declarative output contract + validator.

This is the file you edit first when the problem drops. Describe the required
`output.csv` once here and you get, for free: coercion of model output to legal
values, a hard validator that refuses to ship a malformed submission, and the
column comparators the scorer uses against the labeled sample file.

Every edition so far has scored `output.csv` per row against hidden golden
labels, with closed vocabularies on the categorical columns. A row that is
malformed is a row scored at zero, so validation runs before anything is written.
"""

from __future__ import annotations

import csv
import datetime as _dtmod
import difflib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

ColumnKind = Literal["key", "categorical", "text", "number", "id_list", "boolean", "date"]


@dataclass(frozen=True)
class Column:
    """One column of the required output."""

    name: str
    kind: ColumnKind
    allowed: frozenset[str] | None = None
    fallback: str = ""
    min_words: int = 0
    bounds: tuple[float, float] | None = None
    list_sep: str = ";"
    none_token: str = "none"

    def coerce(self, value: Any) -> str:
        """Repair a value into something legal. Never raises — the validator is
        what refuses; this just gets as close as possible first."""
        text = "" if value is None else str(value).strip()

        if self.kind == "categorical":
            if not self.allowed:
                return text or self.fallback
            if text in self.allowed:
                return text
            lowered = {a.lower(): a for a in self.allowed}
            if text.lower() in lowered:
                return lowered[text.lower()]
            # Nearest legal label beats an out-of-vocabulary string, which scores 0.
            close = difflib.get_close_matches(text.lower(), list(lowered), n=1, cutoff=0.82)
            return lowered[close[0]] if close else self.fallback

        if self.kind == "date":
            # Empty is a meaningful value here: the spec says to leave the
            # date blank when the full amount never becomes safe.
            return text[:10] if text else ""

        if self.kind == "boolean":
            return "true" if text.lower() in {"true", "1", "yes", "y"} else "false"

        if self.kind == "number":
            try:
                # Thousands separators must not silently coerce an amount to
                # zero: "1,234.50" is a real value, and a zero here would ship
                # a wrong recommendation rather than an obvious error.
                number = float(text.replace(",", ""))
            except (TypeError, ValueError):
                number = float(self.fallback or 0.0)
            if self.bounds:
                low, high = self.bounds
                number = max(low, min(high, number))
            if math.isnan(number):
                number = float(self.fallback or 0.0)
            # Plain decimal only: "%g" switches to scientific notation above
            # 1e6, which would corrupt every large-currency amount (IDR).
            rounded = round(number, 2)
            text = f"{rounded:.2f}".rstrip("0").rstrip(".")
            return text or "0"

        if self.kind == "id_list":
            if not text or text.lower() == self.none_token:
                return self.none_token
            parts = [p.strip() for p in text.replace(",", self.list_sep).split(self.list_sep)]
            seen: list[str] = []
            for part in parts:
                if part and part.lower() != self.none_token and part not in seen:
                    seen.append(part)
            return self.list_sep.join(seen) if seen else self.none_token

        return text or self.fallback

    def issues(self, value: str) -> list[str]:
        """Hard problems that should block submission."""
        problems: list[str] = []
        if self.kind == "key" and not value:
            problems.append(f"{self.name}: empty key")
        if self.kind == "categorical" and self.allowed and value not in self.allowed:
            problems.append(f"{self.name}: {value!r} not in allowed set")
        if self.kind == "date" and value:
            try:
                _dtmod.date.fromisoformat(value)
            except ValueError:
                problems.append(f"{self.name}: {value!r} is not YYYY-MM-DD")
        if self.kind == "text":
            if not value:
                problems.append(f"{self.name}: empty")
            elif self.min_words and len(value.split()) < self.min_words:
                problems.append(f"{self.name}: shorter than {self.min_words} words")
        if self.kind == "number" and self.bounds:
            try:
                number = float(value)
            except ValueError:
                problems.append(f"{self.name}: {value!r} is not numeric")
            else:
                low, high = self.bounds
                if not low <= number <= high:
                    problems.append(f"{self.name}: {number} outside [{low}, {high}]")
        return problems


@dataclass
class ValidationReport:
    row_count: int = 0
    expected_count: int = 0
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)
    column_order_ok: bool = True
    row_issues: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            not self.missing_keys
            and not self.unexpected_keys
            and not self.duplicate_keys
            and not self.row_issues
            and self.column_order_ok
            and self.row_count == self.expected_count
        )

    def render(self, limit: int = 12) -> str:
        lines = [
            f"rows: {self.row_count} (expected {self.expected_count})",
            f"column order correct: {self.column_order_ok}",
            f"missing keys: {len(self.missing_keys)}",
            f"unexpected keys: {len(self.unexpected_keys)}",
            f"duplicate keys: {len(self.duplicate_keys)}",
            f"rows with issues: {len(self.row_issues)}",
        ]
        for key, problems in list(self.row_issues.items())[:limit]:
            lines.append(f"  {key}: {'; '.join(problems)}")
        if len(self.row_issues) > limit:
            lines.append(f"  ... and {len(self.row_issues) - limit} more")
        for key in self.missing_keys[:limit]:
            lines.append(f"  MISSING: {key}")
        return "\n".join(lines)


@dataclass(frozen=True)
class OutputSpec:
    """The full required output contract."""

    columns: tuple[Column, ...]
    key_column: str

    @property
    def header(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def coerce_row(self, row: dict[str, Any]) -> dict[str, str]:
        """Produce a legal row in the exact required column order."""
        return {c.name: c.coerce(row.get(c.name)) for c in self.columns}

    def validate(
        self, rows: Sequence[dict[str, str]], expected_keys: Iterable[str]
    ) -> ValidationReport:
        expected = list(expected_keys)
        report = ValidationReport(row_count=len(rows), expected_count=len(expected))

        seen: dict[str, int] = {}
        for row in rows:
            key = str(row.get(self.key_column, "")).strip()
            seen[key] = seen.get(key, 0) + 1
            problems: list[str] = []
            for col in self.columns:
                problems.extend(col.issues(str(row.get(col.name, ""))))
            if problems:
                report.row_issues[key or "<blank>"] = problems

        expected_set = set(expected)
        report.missing_keys = [k for k in expected if k not in seen]
        report.unexpected_keys = [k for k in seen if k not in expected_set]
        report.duplicate_keys = [k for k, n in seen.items() if n > 1]
        if rows:
            report.column_order_ok = list(rows[0].keys()) == self.header
        return report

    def write_csv(self, rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
        """Write in the exact required column order, newline='' so Windows does
        not inject blank lines between records."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.header, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(self.coerce_row(row))
        return target
