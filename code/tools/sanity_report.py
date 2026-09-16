"""Pre-submission sanity and parity report.

Two different questions, which is why this is separate from the test suites:

* **Sanity** — is anything in `output.csv` obviously broken in a way the schema
  cannot see? Degenerate columns, impossible combinations, amounts that are
  absurd for their currency, dates outside the forecast window.
* **Parity** — does the predicted distribution resemble the labelled samples?
  The 25 samples are a small draw, so a difference is a *flag*, not a failure;
  but a column that collapses to one value, or a status that never appears at
  all, usually means a rule is misfiring rather than that the data differs.

Exits non-zero only on hard anomalies. Distribution drift is reported for a
human to judge.

Run:  python tools/sanity_report.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipelines.sept_state import parse_amount, parse_date  # noqa: E402
from pipelines.sept_validate import parse_plan  # noqa: E402
from pipelines.september2026 import Dataset  # noqa: E402

STATUSES = ("affordable_now", "affordable_with_plan", "affordable_later", "not_affordable")
METHODS = ("full_payment", "partial_payment", "installments", "wait", "not_recommended")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def pct(counter: collections.Counter, key: str, total: int) -> float:
    return counter.get(key, 0) / total * 100 if total else 0.0


def parity(samples: list[dict[str, str]], preds: list[dict[str, str]]) -> list[str]:
    """Compare predicted status and method shares with the labelled samples."""
    flags: list[str] = []
    print(f"\n=== PARITY vs the {len(samples)} labelled samples ===\n")
    gold = collections.Counter(s["affordability_status"] for s in samples)
    mine = collections.Counter(p["affordability_status"] for p in preds)
    print(f"{'status':<24}{'gold %':>9}{'pred %':>9}{'delta':>8}")
    print("-" * 50)
    for status in STATUSES:
        g, m = pct(gold, status, len(samples)), pct(mine, status, len(preds))
        flag = "  <<" if abs(g - m) > 25 else ""
        print(f"{status:<24}{g:>8.0f}%{m:>8.0f}%{m - g:>+7.0f}{flag}")
        if abs(g - m) > 25:
            flags.append(f"{status}: {m:.0f}% predicted vs {g:.0f}% in samples")

    gold_m = collections.Counter(s["recommended_payment_method"] for s in samples)
    mine_m = collections.Counter(p["recommended_payment_method"] for p in preds)
    print(f"\n{'method':<24}{'gold %':>9}{'pred %':>9}{'delta':>8}")
    print("-" * 50)
    for method in METHODS:
        g, m = pct(gold_m, method, len(samples)), pct(mine_m, method, len(preds))
        print(f"{method:<24}{g:>8.0f}%{m:>8.0f}%{m - g:>+7.0f}")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Resolved against this file, not the working directory, so the tool behaves
    # the same whether it is run from code/, the repository root, or elsewhere.
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--dataset", default=str(repo / "dataset"))
    parser.add_argument("--out", default=str(repo / "dataset" / "output.csv"))
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    preds = load(Path(args.out))
    requests = {r["request_id"]: r for r in dataset.requests}
    profiles = dataset.profiles
    hard: list[str] = []
    soft: list[str] = []

    print(f"=== SANITY: {len(preds)} predicted rows ===\n")

    # ---- degenerate columns ------------------------------------------------
    for column in ("affordability_status", "recommended_payment_method"):
        distinct = {p[column] for p in preds}
        if len(distinct) == 1:
            hard.append(f"{column} collapsed to a single value: {distinct}")
        missing = set(STATUSES if "status" in column else METHODS) - distinct
        if missing:
            soft.append(f"{column} never predicts: {sorted(missing)}")

    # ---- amounts sane for their currency -----------------------------------
    by_currency: dict[str, list[float]] = collections.defaultdict(list)
    for p in preds:
        request = requests.get(p["request_id"])
        if not request:
            hard.append(f"{p['request_id']} is not in requests.csv")
            continue
        currency = profiles.get(request["user_id"], {}).get("home_currency", "?")
        amount = parse_amount(p["amount_safe_to_pay"]) or 0.0
        requested = parse_amount(request["requested_amount"]) or 0.0
        by_currency[currency].append(amount)
        if amount > requested + 0.02:
            hard.append(f"{p['request_id']}: amount {amount} exceeds requested {requested}")
        if amount < 0:
            hard.append(f"{p['request_id']}: negative amount {amount}")

    print(f"{'currency':<10}{'n':>5}{'zero':>7}{'median':>18}{'max':>20}")
    print("-" * 62)
    for currency, values in sorted(by_currency.items()):
        zeros = sum(1 for v in values if v == 0)
        print(f"{currency:<10}{len(values):>5}{zeros:>7}{statistics.median(values):>18,.2f}"
              f"{max(values):>20,.2f}")

    # ---- dates inside the forecast window ----------------------------------
    for p in preds:
        request = requests.get(p["request_id"])
        if not request:
            continue
        as_of = parse_date(request["request_date"])
        earliest = parse_date(p["earliest_date_for_full_payment"])
        # A fixed day count, not replace(year=+1): that raises on 29 February,
        # which a synthetic request date exposed and the real data never had.
        if earliest and as_of and not (as_of <= earliest <= as_of + dt.timedelta(days=366)):
            hard.append(f"{p['request_id']}: earliest date {earliest} outside a sane window")
        for day, amount in parse_plan(p["payment_plan"]):
            if as_of and day < as_of:
                hard.append(f"{p['request_id']}: payment dated {day} before the request")
            if amount <= 0:
                hard.append(f"{p['request_id']}: non-positive payment {amount}")

    # ---- explanation quality ----------------------------------------------
    lengths = [len(p["decision_explanation"].split()) for p in preds]
    distinct_expl = len({p["decision_explanation"] for p in preds})
    print(f"\nexplanations: {distinct_expl} distinct, "
          f"{min(lengths)}-{max(lengths)} words (median {statistics.median(lengths):.0f})")
    if min(lengths) < 6:
        hard.append("an explanation is shorter than six words")
    if distinct_expl < len(preds) * 0.05:
        soft.append("explanations are nearly all identical")

    # ---- parity against the labelled samples -------------------------------
    if dataset.samples:
        soft.extend(parity(dataset.samples, preds))
    else:
        # With no labelled samples, every predicted share reads as a huge gap
        # against 0% - noise rather than a finding - so don't report it.
        print("\n=== PARITY: skipped, this dataset has no labelled samples ===")

    # ---- verdict -----------------------------------------------------------
    print("\n=== VERDICT ===")
    if hard:
        print(f"\n{len(hard)} HARD anomaly(ies):")
        for item in hard[:15]:
            print(f"  {item}")
    if soft:
        print(f"\n{len(soft)} soft flag(s) — judge these, they are not automatically wrong:")
        for item in soft:
            print(f"  {item}")
    if not hard and not soft:
        suffix = ", distributions consistent with the samples" if dataset.samples else ""
        print(f"\nclean: no anomalies{suffix}")
    elif not hard:
        print("\nno hard anomalies")
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
