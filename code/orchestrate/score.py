"""Scoring against the labeled sample file.

The labeled sample is the only ground truth you get. This module turns it into
a number you can hill-climb on, plus an error dump you can actually read.

Column kinds are scored the way the organizers describe scoring them: exact
match on closed vocabularies, set overlap on ID lists, calibration band on
confidence, and a presence/consistency check on the free-text fields (the
rubric caps justifications that are empty, generic, or contradictory, so this
flags those rather than pretending to grade prose).
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import Column, OutputSpec


@dataclass
class ColumnScore:
    name: str
    kind: str
    scored: int = 0
    correct: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.scored if self.scored else 0.0


@dataclass
class ScoreReport:
    columns: dict[str, ColumnScore]
    matched: int
    missing: list[str]
    mismatches: list[dict[str, Any]]
    confusion: dict[str, Counter]

    @property
    def headline(self) -> float:
        """Mean accuracy across scored columns, ignoring the key column."""
        scored = [c for c in self.columns.values() if c.kind != "key" and c.scored]
        return sum(c.accuracy for c in scored) / len(scored) if scored else 0.0

    def render(self, top_confusions: int = 6) -> str:
        lines = [
            f"matched rows       : {self.matched}",
            f"missing predictions: {len(self.missing)}",
            "",
            f"{'column':<28}{'kind':<14}{'accuracy':>10}",
            "-" * 52,
        ]
        for col in self.columns.values():
            if col.kind == "key":
                continue
            lines.append(f"{col.name:<28}{col.kind:<14}{col.accuracy:>9.1%}")
        lines.append("-" * 52)
        lines.append(f"{'HEADLINE (mean)':<42}{self.headline:>9.1%}")

        for name, counter in self.confusion.items():
            wrong = [(k, v) for k, v in counter.most_common() if k[0] != k[1]]
            if not wrong:
                continue
            lines.append(f"\ntop confusions — {name} (gold -> predicted):")
            for (gold, pred), count in wrong[:top_confusions]:
                lines.append(f"  {gold:>18} -> {pred:<18} x{count}")

        for col in self.columns.values():
            if col.detail:
                lines.append(f"\n{col.name} detail: {col.detail}")
        return "\n".join(lines)

    def write_errors(self, path: str | Path) -> Path:
        """Dump every mismatched row so you can read failures, not just counts."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self.mismatches:
            target.write_text("", encoding="utf-8")
            return target
        fields: list[str] = []
        for row in self.mismatches:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.mismatches)
        return target


def _as_set(value: str, sep: str = ";", none_token: str = "none") -> set[str]:
    parts = {p.strip() for p in (value or "").replace(",", sep).split(sep)}
    parts.discard("")
    if parts == {none_token}:
        return set()
    return {p for p in parts if p.lower() != none_token}


def _score_cell(column: Column, gold: str, pred: str) -> float:
    if column.kind in {"categorical", "boolean"}:
        return 1.0 if gold.strip().lower() == pred.strip().lower() else 0.0

    if column.kind == "id_list":
        gold_set, pred_set = _as_set(gold, column.list_sep), _as_set(pred, column.list_sep)
        if not gold_set and not pred_set:
            return 1.0
        if not gold_set or not pred_set:
            return 0.0
        overlap = len(gold_set & pred_set)
        precision = overlap / len(pred_set)
        recall = overlap / len(gold_set)
        return 0.0 if not overlap else 2 * precision * recall / (precision + recall)

    if column.kind == "number":
        try:
            gold_n, pred_n = float(gold), float(pred)
        except ValueError:
            return 0.0
        # Calibration, not exactness: within 0.1 counts as right.
        return 1.0 if abs(gold_n - pred_n) <= 0.1 else 0.0

    if column.kind == "text":
        # Presence + length band. Prose quality is graded by a model upstream;
        # what we can check deterministically is that it is not empty or stubby.
        words = len(pred.split())
        return 1.0 if words >= max(column.min_words, 4) else 0.0

    return 1.0


def score_predictions(
    spec: OutputSpec,
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    context_columns: Sequence[str] = (),
) -> ScoreReport:
    """Join predictions to gold on the key column and score every column."""
    key = spec.key_column
    pred_by_key = {str(r.get(key, "")).strip(): r for r in predictions}

    columns = {c.name: ColumnScore(c.name, c.kind) for c in spec.columns}
    confusion: dict[str, Counter] = defaultdict(Counter)
    mismatches: list[dict[str, Any]] = []
    missing: list[str] = []
    matched = 0
    confidences: list[tuple[float, float]] = []

    for gold_row in gold:
        gold_key = str(gold_row.get(key, "")).strip()
        pred_row = pred_by_key.get(gold_key)
        if pred_row is None:
            missing.append(gold_key)
            continue
        matched += 1

        row_errors: dict[str, Any] = {key: gold_key}
        wrong_any = False

        for column in spec.columns:
            if column.kind == "key" or column.name not in gold_row:
                continue
            gold_value = str(gold_row.get(column.name, "")).strip()
            pred_value = str(pred_row.get(column.name, "")).strip()
            points = _score_cell(column, gold_value, pred_value)

            col_score = columns[column.name]
            col_score.scored += 1
            col_score.correct += points

            if column.kind == "categorical":
                confusion[column.name][(gold_value, pred_value)] += 1
            if column.kind == "number":
                try:
                    confidences.append((float(gold_value), float(pred_value)))
                except ValueError:
                    pass

            if points < 1.0:
                wrong_any = True
                row_errors[f"gold_{column.name}"] = gold_value
                row_errors[f"pred_{column.name}"] = pred_value

        if wrong_any:
            for extra in context_columns:
                if extra in gold_row:
                    row_errors[extra] = gold_row[extra]
            if "rule" in pred_row:
                row_errors["rule"] = pred_row["rule"]
            mismatches.append(row_errors)

    if confidences:
        gold_conf = [g for g, _ in confidences]
        pred_conf = [p for _, p in confidences]
        for column in spec.columns:
            if column.kind == "number":
                columns[column.name].detail = {
                    "gold_mean": round(statistics.mean(gold_conf), 3),
                    "pred_mean": round(statistics.mean(pred_conf), 3),
                    "mean_abs_error": round(
                        statistics.mean(abs(g - p) for g, p in confidences), 3
                    ),
                    "pred_range": [round(min(pred_conf), 2), round(max(pred_conf), 2)],
                    "gold_range": [round(min(gold_conf), 2), round(max(gold_conf), 2)],
                }
                break

    return ScoreReport(
        columns=columns,
        matched=matched,
        missing=missing,
        mismatches=mismatches,
        confusion=dict(confusion),
    )


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [
            {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]
