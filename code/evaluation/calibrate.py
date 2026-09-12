"""Joint calibration of the forecast's projection rates.

Why this exists
---------------
Tuning the forecast one rule at a time stopped working. An ablation showed that
correcting projected *income* alone regressed accuracy by 5.2 points, because
the remaining errors are **compensating**: projected income is too high and
projected variable spending is also too high, so each was masking the other.
Any correction that moves one side without the other makes the fit worse.

So instead of guessing, fit both together against the 25 labelled samples.

Model
-----
Two scalar multipliers applied at projection time:

    income_scale   - scales every projected recurring inflow
    variable_scale - scales every projected recurring outflow whose cadence is
                     sub-monthly (groceries, transport, dining: the
                     "essential variable spending" the spec says to forecast
                     conservatively). Monthly commitments are contractual
                     amounts and are never scaled.

Objective
---------
Minimise median relative error on `amount_safe_to_pay`, with mean exact-match
across all seven columns as the tie-break. Both are reported so a scale that
improves the headline number while breaking the categorical columns is visible
rather than silently accepted.

Guard against overfitting
-------------------------
25 samples and 2 parameters is a favourable ratio, but the harness still reports
leave-one-out cross-validated error. If LOO error is much worse than in-sample
error, the fit is memorising and should not be shipped.

Usage
-----
    python evaluation/calibrate.py --coarse       # wide sweep
    python evaluation/calibrate.py --refine 1.0 0.9
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pipelines.september2026 as P  # noqa: E402


def evaluate(dataset: P.Dataset, income_scale: float, variable_scale: float) -> dict:
    """Score the whole sample set under one pair of scales."""
    P.INCOME_SCALE = income_scale
    P.VARIABLE_SCALE = variable_scale

    gold = {r["request_id"]: r for r in dataset.samples}
    columns = [c.name for c in P.SPEC.columns if c.kind != "key"]
    errors: list[float] = []
    per_row_exact: list[float] = []
    hits = 0

    for sample in dataset.samples:
        pred = P.solve_request(sample, dataset)
        truth = gold[sample["request_id"]]
        g = float(truth["amount_safe_to_pay"])
        p = float(pred["amount_safe_to_pay"])
        err = abs(p - g) / max(abs(g), 1.0) * 100
        errors.append(err)

        row_hits = 0
        for column in columns:
            if column == "amount_safe_to_pay":
                ok = err < 1
            else:
                ok = str(truth.get(column, "")).strip() == str(pred.get(column, "")).strip()
            row_hits += ok
        hits += row_hits
        per_row_exact.append(row_hits / len(columns))

    return {
        "median_err": statistics.median(errors),
        "mean_err": statistics.mean(errors),
        "mean_exact": hits / (len(dataset.samples) * len(columns)),
        "within_5pct": sum(1 for e in errors if e < 5),
        "errors": errors,
        "per_row_exact": per_row_exact,
    }


def sweep(dataset: P.Dataset, incomes: list[float], variables: list[float]) -> list[tuple]:
    results = []
    for income in incomes:
        for variable in variables:
            score = evaluate(dataset, income, variable)
            results.append((-score["mean_exact"], score["median_err"], income, variable, score))
    results.sort(key=lambda r: (r[0], r[1]))
    return results


def leave_one_out(dataset: P.Dataset, incomes: list[float], variables: list[float]) -> float:
    """Median held-out error when the scales are fitted without each sample."""
    held: list[float] = []
    samples = list(dataset.samples)
    for index in range(len(samples)):
        dataset.samples = samples[:index] + samples[index + 1:]
        best = sweep(dataset, incomes, variables)[0]
        dataset.samples = [samples[index]]
        held.append(evaluate(dataset, best[2], best[3])["median_err"])
    dataset.samples = samples
    return statistics.median(held)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="../dataset")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--refine", nargs=2, type=float, metavar=("INCOME", "VARIABLE"))
    parser.add_argument("--loo", action="store_true", help="leave-one-out cross-validation")
    args = parser.parse_args()

    dataset = P.Dataset(args.dataset)

    if args.refine:
        centre_i, centre_v = args.refine
        incomes = [round(centre_i + d, 3) for d in (-0.06, -0.03, 0, 0.03, 0.06)]
        variables = [round(centre_v + d, 3) for d in (-0.06, -0.03, 0, 0.03, 0.06)]
    else:
        incomes = [round(0.70 + 0.05 * i, 2) for i in range(13)]      # 0.70 .. 1.30
        variables = [round(0.60 + 0.05 * i, 2) for i in range(15)]    # 0.60 .. 1.30

    baseline = evaluate(dataset, 1.0, 1.0)
    print(f"baseline (1.00, 1.00): median_err={baseline['median_err']:.1f}%  "
          f"mean_exact={baseline['mean_exact']:.1%}  <5%={baseline['within_5pct']}/25\n")

    results = sweep(dataset, incomes, variables)
    print(f"{'income':>8}{'variable':>10}{'medErr':>9}{'meanErr':>10}{'exact':>9}{'<5%':>6}")
    print("-" * 52)
    for neg_exact, med, income, variable, score in results[:12]:
        print(f"{income:>8.2f}{variable:>10.2f}{med:>8.1f}%{score['mean_err']:>9.1f}%"
              f"{-neg_exact:>8.1%}{score['within_5pct']:>5}")

    best = results[0]
    print(f"\nBEST: income_scale={best[2]:.2f}  variable_scale={best[3]:.2f}")
    print(f"  mean_exact {baseline['mean_exact']:.1%} -> {-best[0]:.1%}")
    print(f"  median_err {baseline['median_err']:.1f}% -> {best[1]:.1f}%")

    if args.loo:
        loo = leave_one_out(dataset, incomes, variables)
        print(f"\nleave-one-out median error: {loo:.1f}%  (in-sample {best[1]:.1f}%)")
        if loo > best[1] * 1.8:
            print("  WARNING: held-out error much worse than in-sample - this is "
                  "overfitting the 25 samples. Do not ship these scales.")
        else:
            print("  Held-out error is comparable: the fit generalises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
