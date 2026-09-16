"""Buy or Wait? — entry point.

    python main.py run          # produce dataset/output.csv for all requests
    python main.py score        # grade the pipeline against sample_requests.csv
    python main.py validate     # check an existing output.csv
    python main.py extract      # (re)build model extractions — needs an API key
    python main.py calibrate    # refit the projection scales
    python main.py test         # unit, guard and invariant tests
    python main.py sanity       # sanity and parity report on output.csv
    python main.py mutate       # mutation testing: are the tests load-bearing?
    python main.py verify       # the full pre-submission gate (same as CI)

`run` needs no API key: every model extraction is cached under `extracted/`
and committed with the solution, so the decision pipeline is fully reproducible
offline. `extract` is the only command that calls a model.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipelines.sept_validate import validate_all  # noqa: E402
from pipelines.september2026 import SPEC, Dataset, fallback_row, solve_request  # noqa: E402

# Resolved against this file, not the caller's working directory: the defaults
# must mean the same thing whether the command is run from code/ or the repo
# root, otherwise `--out ../dataset/output.csv` writes outside the repository.
DEFAULT_DATASET = str(HERE.parent / "dataset")
DEFAULT_OUT = str(HERE.parent / "dataset" / "output.csv")
DEFAULT_ERRORS = str(HERE / "evaluation" / "errors.csv")


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
    if not rows:
        # Synthetic data has no independent ground truth, so there is nothing to
        # grade against. Say so plainly rather than print a meaningless 100%.
        print("no labelled samples in this dataset - nothing to score against.")
        print("MEAN n/a")
        return 0
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
    # Dump every mismatched row. Reading failures is what moves the score; a
    # column percentage only says that something, somewhere, is wrong.
    errors_path = Path(args.errors)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches: list[dict] = []
    for pred in preds:
        truth = gold[pred["request_id"]]
        row: dict = {"request_id": pred["request_id"]}
        wrong = False
        for column in columns:
            g, p = str(truth.get(column, "")).strip(), str(pred.get(column, "")).strip()
            if column == "amount_safe_to_pay":
                err = abs(float(g) - float(p)) / max(abs(float(g)), 1.0) * 100
                row["amount_error_pct"] = round(err, 2)
                if err >= 1:
                    wrong = True
                    row["gold_amount_safe_to_pay"] = g
                    row["pred_amount_safe_to_pay"] = p
            elif g != p:
                wrong = True
                row[f"gold_{column}"] = g
                row[f"pred_{column}"] = p
        if wrong:
            request = next(r for r in rows if r["request_id"] == pred["request_id"])
            row["user_id"] = request["user_id"]
            row["request_type"] = request["request_type"]
            mismatches.append(row)

    if mismatches:
        fields: list[str] = []
        for row in mismatches:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with errors_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(mismatches)
    print(f"\nwrote {errors_path} ({len(mismatches)} mismatched rows)")

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
    # Pin the working directory to this file's own. Sub-scripts resolve the
    # dataset relative to code/, so inheriting an arbitrary caller cwd makes
    # `verify` pass or fail depending on where it was invoked from.
    return subprocess.call([sys.executable, str(HERE / script), *extra], cwd=HERE)


def missing_key_message(environ: dict[str, str] | None = None) -> str | None:
    """Explain, before any work starts, why `extract` cannot call a model."""
    import os

    env = os.environ if environ is None else environ
    if env.get("ORCHESTRATE_DRY_RUN", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    if env.get("OPENAI_API_KEY", "").strip():
        return None
    return ("extract calls a model and needs OPENAI_API_KEY. Copy code/.env.example to "
            "code/.env and set it, or set ORCHESTRATE_DRY_RUN=1 to serve from cache only. "
            "`python main.py run` needs no key.")


def cmd_extract(args) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(HERE / ".env")
    except ImportError:  # pragma: no cover - python-dotenv is a core requirement
        pass
    problem = missing_key_message()
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    rc = _run("tools/run_extract.py")
    return rc or _run("tools/run_descriptions.py")


def cmd_calibrate(args) -> int:
    return _run("evaluation/calibrate.py", "--dataset", args.dataset, "--loo")


def cmd_test(args) -> int:
    for script in ("tests/test_harness.py", "tests/test_resilience.py",
                   "tests/test_invariants.py"):
        rc = _run(script)
        if rc:
            return rc
    return 0


def cmd_mutate(args) -> int:
    """Slow: re-runs the whole suite once per mutant."""
    return _run("tests/test_mutation.py")


def cmd_sanity(args) -> int:
    return _run("tools/sanity_report.py", "--dataset", args.dataset, "--out", args.out)


def cmd_verify(args) -> int:
    """The full pre-submission gate, in the order a failure is cheapest to find.

    Same sequence as CI, so a green local run and a green pipeline mean the
    same thing.
    """
    steps = [
        ("unit tests", lambda: _run("tests/test_harness.py")),
        ("guard tests", lambda: _run("tests/test_resilience.py")),
        ("invariants", lambda: _run("tests/test_invariants.py")),
        ("produce output.csv", lambda: cmd_run(args)),
        ("sanity and parity", lambda: cmd_sanity(args)),
        ("accuracy", lambda: cmd_score(args)),
        ("package", lambda: _run("tools/package.py")),
    ]
    rule = "=" * 70
    for name, step in steps:
        print(f"\n{rule}\n  {name}\n{rule}")
        rc = step()
        if rc:
            print(f"\nVERIFY FAILED at: {name}")
            return rc
    print(f"\n{rule}\n  VERIFY PASSED — all checks green\n{rule}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="dataset directory (default: ../dataset)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="predictions path (default: ../dataset/output.csv)")
    parser.add_argument("--errors", default=DEFAULT_ERRORS,
                        help="where `score` dumps mismatched rows")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_text in (
        ("run", cmd_run, "produce output.csv for every request"),
        ("score", cmd_score, "grade against the labelled samples"),
        ("validate", cmd_validate, "check an existing output.csv"),
        ("extract", cmd_extract, "rebuild model extractions (needs an API key)"),
        ("calibrate", cmd_calibrate, "refit projection scales"),
        ("test", cmd_test, "run unit, guard and invariant tests"),
        ("mutate", cmd_mutate, "mutation testing: are the tests load-bearing?"),
        ("sanity", cmd_sanity, "sanity and parity report on output.csv"),
        ("verify", cmd_verify, "full pre-submission gate (same sequence as CI)"),
    ):
        sub.add_parser(name, help=help_text).set_defaults(func=fn)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
