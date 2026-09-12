"""Buy or Wait? — entry point.

    python main.py run          # produce dataset/output.csv for all requests
    python main.py score        # grade the pipeline against sample_requests.csv
    python main.py validate     # check an existing output.csv
    python main.py extract      # (re)build model extractions — needs an API key
    python main.py calibrate    # refit the projection scales
    python main.py test         # unit + invariant tests

`run` needs no API key: every model extraction is cached under `extracted/`
and committed with the solution, so the decision pipeline is fully reproducible
offline. `extract` is the only command that calls a model.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipelines.september2026 import SPEC, Dataset, fallback_row, solve_request  # noqa: E402
from pipelines.sept_validate import validate_all  # noqa: E402

DEFAULT_DATASET = "../dataset"


def _solve_all(dataset: Dataset, rows) -> tuple[list[dict], list[tuple[str, str]]]:
    preds, failures = [], []
    for row in rows:
        try:
            preds.append(solve_request(row, dataset))
        except Exception as exc:  # noqa: BLE001 — one bad row must not end the run
            failures.append((row.get("request_id", "?"), f"{type(exc).__name__}: {exc}"))
            preds.append(fallback_row(row, exc))
    return preds, failures


def cmd_run(args) -> int:
    dataset = Dataset(args.dataset)
    rows = dataset.requests
    preds, failures = _solve_all(dataset, rows)
    print(f"solved {len(preds)} request(s), {len(failures)} failure(s)")
    for rid, err in failures[:5]:
        print(f"  FAILED {rid}: {err}")

    report = SPEC.validate([SPEC.coerce_row(p) for p in preds],
                           [r["request_id"] for r in rows])
    print("\n--- output contract ---")
    print(report.render())

    cross = validate_all(rows, preds, profiles=dataset.profiles,
                         options_by_request=dataset.options)
    bad = [c for c in cross if not c.ok]
    print(f"\n--- cross-field validation ---\nrows with violations: {len(bad)}")
    for c in bad[:10]:
        print(f"  {c.request_id}: {'; '.join(c.violations)}")

    SPEC.write_csv(preds, args.out)
    print(f"\nwrote {args.out}")
    for column in ("affordability_status", "recommended_payment_method"):
        print(f"{column}: {dict(collections.Counter(p[column] for p in preds))}")
    return 0 if report.ok and not bad else 1


def cmd_score(args) -> int:
    dataset = Dataset(args.dataset)
    rows = dataset.samples
    preds, failures = _solve_all(dataset, rows)
    gold = {r["request_id"]: r for r in rows}
    columns = [c.name for c in SPEC.columns if c.kind != "key"]

    hits: collections.Counter = collections.Counter()
    errors: list[float] = []
    for pred in preds:
        truth = gold[pred["request_id"]]
        for column in columns:
            g, p = str(truth.get(column, "")).strip(), str(pred.get(column, "")).strip()
            if column == "amount_safe_to_pay":
                err = abs(float(g) - float(p)) / max(abs(float(g)), 1.0) * 100
                errors.append(err)
                hits[column] += err < 1
            else:
                hits[column] += g == p
    n = len(preds)
    print(f"\n{'column':<34}{'exact':>12}")
    print("-" * 46)
    for column in columns:
        print(f"{column:<34}{hits[column]:>4}/{n:<4}{hits[column] / n:>6.0%}")
    print("-" * 46)
    print(f"{'MEAN':<34}{statistics.mean(hits[c] / n for c in columns):>11.1%}")
    print(f"amount_safe_to_pay median error: {statistics.median(errors):.1f}%")
    return 0


def cmd_validate(args) -> int:
    dataset = Dataset(args.dataset)
    from orchestrate.schema import OutputSpec  # noqa: F401  (kept explicit for clarity)
    import csv

    with open(args.out, newline="", encoding="utf-8-sig") as handle:
        preds = [dict(r) for r in csv.DictReader(handle)]
    report = SPEC.validate(preds, [r["request_id"] for r in dataset.requests])
    print(report.render())
    cross = validate_all(dataset.requests, preds, profiles=dataset.profiles,
                         options_by_request=dataset.options)
    bad = [c for c in cross if not c.ok]
    print(f"\ncross-field violations: {len(bad)}")
    for c in bad[:10]:
        print(f"  {c.request_id}: {'; '.join(c.violations)}")
    return 0 if report.ok and not bad else 1


def _run(script: str, *extra: str) -> int:
    return subprocess.call([sys.executable, str(HERE / script), *extra])


def cmd_extract(args) -> int:
    rc = _run("tools/run_extract.py")
    return rc or _run("tools/run_descriptions.py")


def cmd_calibrate(args) -> int:
    return _run("evaluation/calibrate.py", "--dataset", args.dataset, "--loo")


def cmd_test(args) -> int:
    rc = _run("tests/test_harness.py")
    return rc or _run("tests/test_invariants.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"dataset directory (default {DEFAULT_DATASET})")
    parser.add_argument("--out", default=f"{DEFAULT_DATASET}/output.csv",
                        help="predictions path")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_text in (
        ("run", cmd_run, "produce output.csv for every request"),
        ("score", cmd_score, "grade against the labelled samples"),
        ("validate", cmd_validate, "check an existing output.csv"),
        ("extract", cmd_extract, "rebuild model extractions (needs an API key)"),
        ("calibrate", cmd_calibrate, "refit projection scales"),
        ("test", cmd_test, "run unit and invariant tests"),
    ):
        sub.add_parser(name, help=help_text).set_defaults(func=fn)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
