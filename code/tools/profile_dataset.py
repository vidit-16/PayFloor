"""Dataset profiler — run this in the first ten minutes, before writing agent code.

    python tools/profile_dataset.py --dataset path/to/dataset

It answers, without you reading a single CSV by hand:

* which file is the input, which is the labeled sample, which is the template
* what the label vocabularies and their distributions are
* what register the golden free-text fields are written in (word band, and
  whether they repeat verbatim — they did in August)
* what band the golden confidence column sits in, tiered by the action column
* **which input rows are minimal pairs** — identical text needing different
  labels from context alone
* which rows carry prompt-injection or social-engineering markers
* which context tables join to the input, and on which column
* which media files are reused across rows (cache them once, not per row)
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows consoles default to cp1252; force UTF-8 so output is never mangled.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Lightweight adversarial-marker scan, kept local so the tool has no
# dependency on the decision pipeline.
import re as _re

_MARKERS = (r'\bignore (previous|prior|above)\b', r'\bsystem note\b',
            r'\b(set|mark|classify)\s+\w+\s*(as|=)\b', r'\boverride\b')


def matched_signals(text):
    low = (text or '').lower()
    return {'injection': any(_re.search(m, low) for m in _MARKERS)}

MAX_CATEGORICAL = 25
TEXTY_MIN_MEAN_WORDS = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def classify_column(rows: list[dict[str, str]], column: str) -> str:
    values = [r.get(column, "") for r in rows]
    non_empty = [v for v in values if v]
    if not non_empty:
        return "empty"
    distinct = len(set(non_empty))
    mean_words = statistics.mean(len(v.split()) for v in non_empty)

    numeric = 0
    for v in non_empty:
        try:
            float(v)
            numeric += 1
        except ValueError:
            break
    if numeric == len(non_empty):
        return "numeric"
    if distinct == len(values) and mean_words <= 3:
        return "identifier"
    if mean_words >= TEXTY_MIN_MEAN_WORDS:
        return "text"
    if distinct <= MAX_CATEGORICAL:
        return "categorical"
    return "identifier"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def profile_file(path: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    print(f"\n--- {path.name}  ({len(rows)} rows x {len(rows[0]) if rows else 0} cols) ---")
    kinds: dict[str, str] = {}
    if not rows:
        return kinds
    for column in rows[0]:
        kind = classify_column(rows, column)
        kinds[column] = kind
        values = [r.get(column, "") for r in rows]
        non_empty = [v for v in values if v]
        blank = len(values) - len(non_empty)
        note = ""

        if kind == "categorical":
            counts = Counter(non_empty)
            note = "  ".join(f"{k}={v}" for k, v in counts.most_common(8))
            if len(counts) > 8:
                note += f"  (+{len(counts) - 8} more)"
        elif kind == "numeric":
            numbers = [float(v) for v in non_empty]
            note = (
                f"min={min(numbers):g} max={max(numbers):g} "
                f"mean={statistics.mean(numbers):.3f} distinct={len(set(numbers))}"
            )
        elif kind == "text":
            words = [len(v.split()) for v in non_empty]
            repeats = Counter(non_empty)
            dupes = sum(c - 1 for c in repeats.values() if c > 1)
            note = (
                f"words min={min(words)} max={max(words)} mean={statistics.mean(words):.1f}"
                f" | distinct={len(repeats)}"
            )
            if dupes:
                note += f" | VERBATIM REPEATS: {dupes}"
        elif kind == "identifier":
            note = f"distinct={len(set(non_empty))}"

        print(f"  {column:<32} {kind:<12} blank={blank:<4} {note}")
    return kinds


def report_text_templates(rows: list[dict[str, str]], column: str) -> None:
    counts = Counter(r.get(column, "") for r in rows if r.get(column))
    repeated = [(t, c) for t, c in counts.most_common() if c > 1]
    if not repeated:
        return
    print(f"\n  repeated {column!r} values (the golden set uses templates - match this register):")
    for text, count in repeated[:10]:
        print(f"    x{count}  {text[:100]}")


def report_confidence_tiers(
    rows: list[dict[str, str]], number_col: str, group_col: str
) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            groups[row.get(group_col, "?")].append(float(row.get(number_col, "")))
        except ValueError:
            continue
    if not groups:
        return
    print(f"\n  {number_col} banded by {group_col}:")
    for name, values in sorted(groups.items()):
        print(
            f"    {name:<20} n={len(values):<4} min={min(values):.2f} "
            f"max={max(values):.2f} mean={statistics.mean(values):.3f}"
        )


def report_minimal_pairs(rows: list[dict[str, str]], column: str, key: str) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(column, "").strip()
        if value:
            groups[value].append(row)
    dupes = {v: g for v, g in groups.items() if len(g) > 1}
    if not dupes:
        print(f"  no exact duplicates in {column!r}")
        return
    covered = sum(len(g) for g in dupes.values())
    print(
        f"  *** {len(dupes)} distinct {column!r} value(s) repeat across {covered} rows "
        f"of {len(rows)} - these are MINIMAL PAIRS ***"
    )
    varying = [c for c in rows[0] if c != column]
    for value, group in sorted(dupes.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"\n    x{len(group)}  {value[:90]!r}")
        differing = [
            c for c in varying if len({r.get(c, "") for r in group}) > 1 and c != key
        ]
        print(f"      differs on: {', '.join(differing) or '(nothing - pure repetition)'}")
        for row in group:
            shown = {c: row.get(c, "") for c in [key] + differing[:4]}
            print(f"      {shown}")


def report_adversarial(rows: list[dict[str, str]], text_columns: list[str], key: str) -> None:
    hits: list[tuple[str, str, str]] = []
    for row in rows:
        blob = " ".join(row.get(c, "") for c in text_columns)
        signals = matched_signals(blob)
        fired = [name for name, on in signals.items() if on]
        if fired:
            hits.append((row.get(key, "?"), ",".join(fired), blob[:150]))
    print(f"  rows carrying adversarial markers: {len(hits)} of {len(rows)}")
    for row_key, fired, snippet in hits[:15]:
        print(f"    {row_key:<12} [{fired:<26}] {snippet!r}")


def report_joins(
    input_rows: list[dict[str, str]], tables: dict[str, list[dict[str, str]]], input_name: str
) -> None:
    if not input_rows:
        return
    print("  columns in the input that join to other tables:")
    found = False
    for column in input_rows[0]:
        values = {r.get(column, "") for r in input_rows if r.get(column)}
        if not values or len(values) == len(input_rows):
            continue  # unique per row => the key, not a foreign key
        for table_name, rows in tables.items():
            if table_name == input_name or not rows:
                continue
            for other in rows[0]:
                other_values = {r.get(other, "") for r in rows if r.get(other)}
                if not other_values:
                    continue
                overlap = values & other_values
                if len(overlap) / len(values) >= 0.5:
                    found = True
                    print(
                        f"    {column:<24} -> {table_name}.{other:<24} "
                        f"covers {len(overlap)}/{len(values)} distinct values"
                    )
    if not found:
        print("    (none detected - check the problem statement for the join keys)")


def report_media(tables: dict[str, list[dict[str, str]]], input_rows: list[dict[str, str]]) -> None:
    manifests = {
        name: rows
        for name, rows in tables.items()
        if rows and any(re.search(r"path|file|url", c, re.I) for c in rows[0])
    }
    if not manifests:
        return
    for name, rows in manifests.items():
        path_col = next(c for c in rows[0] if re.search(r"path|file|url", c, re.I))
        id_col = next((c for c in rows[0] if c != path_col), None)
        print(f"  {name}: {len(rows)} entries (id={id_col}, path={path_col})")
        if id_col and input_rows:
            ids = {r.get(id_col, "") for r in rows}
            for column in input_rows[0]:
                used = Counter(
                    r.get(column, "") for r in input_rows if r.get(column, "") in ids
                )
                if used:
                    reused = {k: v for k, v in used.items() if v > 1}
                    print(
                        f"    referenced by input.{column}: {sum(used.values())} rows, "
                        f"{len(used)} distinct"
                    )
                    if reused:
                        print(
                            f"    REUSED across rows (cache per media id!): {reused}"
                        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="dataset directory")
    parser.add_argument("--input", help="input CSV filename (auto-detected if omitted)")
    parser.add_argument("--sample", help="labeled sample CSV filename (auto-detected)")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    paths = sorted(root.glob("*.csv"))
    tables = {p.stem: read_csv(p) for p in paths}

    banner(f"FILES IN {root}")
    for path in paths:
        print(f"  {path.name:<36} {len(tables[path.stem]):>6} rows")
    media_dirs = [d for d in root.iterdir() if d.is_dir()]
    for directory in media_dirs:
        files = [f for f in directory.rglob("*") if f.is_file()]
        print(f"  {directory.name + '/':<36} {len(files):>6} files")

    # Identify the three special files by shape.
    sample_name = args.sample and Path(args.sample).stem
    input_name = args.input and Path(args.input).stem
    template_name = None

    if not sample_name:
        candidates = [(n, len(r[0]) if r else 0) for n, r in tables.items() if "sample" in n]
        sample_name = max(candidates, key=lambda c: c[1])[0] if candidates else None
    if not template_name:
        for name, rows in tables.items():
            if "output" in name and (not rows or all(not v for v in rows[0].values() if v != list(rows[0].values())[0])):
                template_name = name
    if not input_name:
        # The template names exactly the rows needing predictions, so the input
        # is whichever table carries those same keys. Column-overlap alone is
        # ambiguous (a history table usually has the identical schema).
        template_rows = tables.get(template_name or "", [])
        template_keys = (
            {r.get(next(iter(template_rows[0])), "") for r in template_rows}
            if template_rows
            else set()
        )
        best, best_score = None, 0.0
        for name, rows in tables.items():
            if name in {sample_name, template_name} or not rows:
                continue
            keys = {r.get(next(iter(rows[0])), "") for r in rows}
            if template_keys:
                score = len(keys & template_keys) / len(template_keys)
            elif sample_name and tables.get(sample_name):
                sample_cols = set(tables[sample_name][0])
                cols = set(rows[0])
                score = len(cols & sample_cols) / len(sample_cols) if cols < sample_cols else 0.0
            else:
                score = 0.0
            if score > best_score:
                best, best_score = name, score
        input_name = best

    print(f"\n  detected input   : {input_name}")
    print(f"  detected sample  : {sample_name}")
    print(f"  detected template: {template_name}")

    banner("COLUMN PROFILE - EVERY FILE")
    kinds_by_file: dict[str, dict[str, str]] = {}
    for path in paths:
        kinds_by_file[path.stem] = profile_file(path, tables[path.stem])

    if sample_name and tables.get(sample_name):
        sample_rows = tables[sample_name]
        input_cols = set(tables[input_name][0]) if input_name and tables.get(input_name) else set()
        label_cols = [c for c in sample_rows[0] if c not in input_cols]

        banner(f"GOLDEN LABEL ANALYSIS - {sample_name}")
        print(f"  label columns: {label_cols}")
        kinds = kinds_by_file[sample_name]

        for column in label_cols:
            kind = kinds.get(column)
            if kind == "text":
                report_text_templates(sample_rows, column)
            if kind == "numeric":
                group = next(
                    (c for c in label_cols if kinds.get(c) == "categorical"), None
                )
                if group:
                    report_confidence_tiers(sample_rows, column, group)

        cats = [c for c in label_cols if kinds.get(c) == "categorical"]
        if len(cats) >= 2:
            print(f"\n  joint distribution {cats[0]} x {cats[1]}:")
            joint = Counter((r.get(cats[0], ""), r.get(cats[1], "")) for r in sample_rows)
            for (a, b), count in sorted(joint.items()):
                print(f"    {a:<16} {b:<20} {count}")

    if input_name and tables.get(input_name):
        input_rows = tables[input_name]
        key = next(iter(input_rows[0]))
        kinds = kinds_by_file[input_name]
        text_cols = [c for c, k in kinds.items() if k == "text"]

        banner("MINIMAL PAIRS IN THE INPUT (identical text, different context)")
        for column in text_cols:
            print(f"\n  column: {column}")
            report_minimal_pairs(input_rows, column, key)

        banner("ADVERSARIAL / INJECTION SCAN")
        report_adversarial(input_rows, text_cols, key)

        banner("CONTEXT JOINS")
        report_joins(input_rows, tables, input_name)

        banner("MEDIA")
        report_media(tables, input_rows)

    banner("NEXT STEPS")
    print(
        "  1. Copy the label vocabularies above into an OutputSpec in orchestrate/schema.py\n"
        "  2. Write reason templates matching the golden register (word band + repeats)\n"
        "  3. Set confidence tiers from the banded ranges above\n"
        "  4. Encode the minimal-pair distinguishers as deterministic policy rules\n"
        "  5. Make every adversarial row a safety-tier rule, non-overridable\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
