"""Buy or Wait? — the wired pipeline.

Per request: reconstruct financial state, apply model-extracted amendments,
forecast 90 days, enumerate plans, rank, emit the row.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchestrate.schema import Column, OutputSpec

from .sept_solver import (
    METHOD_FULL,
    METHOD_INSTALL,
    METHOD_PARTIAL,
    STATUS_NOT,
    Candidate,
    PaymentOption,
    build_candidates,
    classify,
    earliest_full_payment_date,
    explain,
    load_options,
    max_safe_today,
    select,
)
from .sept_state import (
    Adjustment,
    Event,
    Forecast,
    RateTable,
    Recurring,
    build_events,
    parse_amount,
    parse_date,
    split_list,
)

HORIZON = 90

SPEC = OutputSpec(
    key_column="request_id",
    columns=(
        Column("request_id", "key"),
        Column("amount_safe_to_pay", "number", fallback="0"),
        Column(
            "affordability_status", "categorical",
            allowed=frozenset({"affordable_now", "affordable_with_plan",
                               "affordable_later", "not_affordable"}),
            fallback="not_affordable",
        ),
        Column(
            "recommended_payment_method", "categorical",
            allowed=frozenset({"full_payment", "partial_payment", "installments",
                               "wait", "not_recommended"}),
            fallback="not_recommended",
        ),
        Column("payment_plan", "text", fallback="none"),
        Column("earliest_date_for_full_payment", "date", fallback=""),
        Column("spending_changes_needed", "text", fallback="none"),
        Column("decision_explanation", "text", min_words=6),
    ),
)


def add_months(day: dt.date, n: int) -> dt.date:
    year, month = divmod(day.month - 1 + n, 12)
    year += day.year
    month += 1
    return dt.date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


class Dataset:
    """All CSVs plus the cached model extractions, indexed for per-request use."""

    def __init__(self, root: str | Path, extracted: str | Path = "extracted") -> None:
        self.root = Path(root)
        load = lambda name: list(  # noqa: E731
            csv.DictReader(open(self.root / name, newline="", encoding="utf-8-sig"))
        )
        self.requests = load("requests.csv")
        self.samples = load("sample_requests.csv")
        self.profiles = {p["user_id"]: p for p in load("financial_profiles.csv")}
        self.rates = RateTable(load("exchange_rates.csv"))

        self.events: dict[str, list[Mapping[str, str]]] = defaultdict(list)
        for row in load("financial_events.csv"):
            self.events[row["user_id"]].append(row)

        self.options: dict[str, list[PaymentOption]] = defaultdict(list)
        for row in load("request_payment_options.csv"):
            self.options[row["request_id"]].append(row)  # type: ignore[arg-type]
        self.options = {k: load_options(v) for k, v in self.options.items()}  # type: ignore[arg-type]

        self.messages: dict[str, list[Mapping[str, str]]] = defaultdict(list)
        for row in load("messages.csv"):
            self.messages[row["user_id"]].append(row)

        self.images = load("images.csv")

        ex = Path(extracted)
        self.message_facts: dict[str, Any] = (
            json.loads((ex / "messages.json").read_text(encoding="utf-8"))
            if (ex / "messages.json").exists() else {}
        )
        self.description_facts: dict[str, Any] = (
            json.loads((ex / "descriptions.json").read_text(encoding="utf-8"))
            if (ex / "descriptions.json").exists() else {}
        )
        self.image_facts: dict[str, Any] = (
            json.loads((ex / "images.json").read_text(encoding="utf-8"))
            if (ex / "images.json").exists() else {}
        )
        self.amount_overrides = {
            row["related_event_id"]: self.image_facts[row["image_id"]]["amount"]
            for row in self.images
            if row.get("related_event_id")
            and self.image_facts.get(row["image_id"], {}).get("amount") is not None
        }


# --------------------------------------------------------------------------- #
# Recurrence + amendments
# --------------------------------------------------------------------------- #


def infer_recurring(
    events: Sequence[Event],
    as_of: dt.date,
    description_facts: Mapping[str, Any] | None = None,
) -> list[Recurring]:
    """Category-level recurrence, corrected by what the descriptions say.

    Grouping is by category (see DESIGN.md 2), but the description carries
    semantics the columns do not, and ignoring it is what made the forecast
    wrong for any user whose circumstances changed:

    * `final` - "Final employer payroll" means the income stops. Projecting it
      forward invents months of salary the user will never receive.
    * `resumes` - a pause (parental leave) inflates the observed median gap, so
      the cadence must come from the commitment, not the gap.
    * `superseded` - authorisations and reversals are provisional records and
      must never seed a recurring flow.
    """
    facts = description_facts or {}
    groups: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.date > as_of:
            continue
        if facts.get(event.description, {}).get("continuity") == "superseded":
            continue
        groups[event.category].append(event)

    out: list[Recurring] = []
    for category, group in groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda e: e.date)
        last = group[-1]

        # The flow has explicitly ended - do not project it at all.
        if facts.get(last.description, {}).get("continuity") == "final":
            continue

        gaps = [(group[i + 1].date - group[i].date).days for i in range(len(group) - 1)]
        if not gaps:
            continue
        median_gap = statistics.median(gaps)

        # A commitment interrupted by leave shows an inflated gap. Snap any
        # near-multiple of a month back to monthly.
        is_commitment = facts.get(last.description, {}).get("is_commitment", False)
        resumes = any(
            facts.get(e.description, {}).get("continuity") == "resumes" for e in group
        )
        if (is_commitment or resumes) and median_gap > 32:
            if abs(median_gap / 30.44 - round(median_gap / 30.44)) < 0.25:
                median_gap = 30.44

        if median_gap > 45:
            continue

        out.append(
            Recurring(
                key=(category, category),
                period_days=max(1, round(median_gap)),
                amount=statistics.median([e.signed for e in group]),
                last_date=last.date,
                category=category,
                description=last.description,
                flexibility=last.flexibility,
                minimum_allowed=last.minimum_allowed,
                event_ids=[e.event_id for e in group],
                representative_id=last.event_id,
            )
        )
    return out


def apply_amendments(
    recurring: list[Recurring],
    dataset: Dataset,
    user_id: str,
    home_currency: str,
    as_of: dt.date,
) -> tuple[list[Recurring], list[tuple[dt.date, float]]]:
    """Fold confirmed message facts into the projection.

    Only confirmed, quantified facts are used. Anything pending, estimated or
    conditional is dropped rather than guessed - counting an unapproved bonus as
    income is exactly what turns an unsafe plan into a apparently safe one.
    """
    horizon = as_of + dt.timedelta(days=HORIZON)
    one_offs: list[tuple[dt.date, float]] = []
    by_category = {r.category: r for r in recurring}

    for row in dataset.messages.get(user_id, []):
        facts = dataset.message_facts.get(row.get("message_id", ""))
        if not facts or not facts.get("concerns_money") or not facts.get("is_confirmed"):
            continue
        amount = facts.get("amount")
        if amount is None or facts.get("change_type") in {"none", ""}:
            continue

        effective = parse_date(facts.get("effective_date", "")) or parse_date(
            (row.get("sent_at") or "")[:10]
        )
        category = (facts.get("category") or "other").strip().lower()
        converted = dataset.rates.convert(
            float(amount), (facts.get("currency") or home_currency), home_currency,
            effective or as_of,
        )

        if facts.get("change_type") == "cancellation":
            flow = by_category.get(category)
            if flow:
                recurring = [r for r in recurring if r is not flow]
                by_category.pop(category, None)
            continue

        if facts.get("is_recurring") and facts.get("supersedes_history"):
            flow = by_category.get(category)
            if flow is not None:
                sign = 1.0 if flow.amount > 0 else -1.0
                flow.amount = sign * abs(converted)
            continue

        if not facts.get("is_recurring") and effective and as_of < effective <= horizon:
            inflow = category in {"salary", "income", "bonus"} or facts.get("change_type") == "refund"
            one_offs.append((effective, converted if inflow else -converted))

    return recurring, one_offs


def project(
    *,
    as_of: dt.date,
    start_balance: float,
    minimum_balance: float,
    events: Sequence[Event],
    recurring: Sequence[Recurring],
    one_offs: Sequence[tuple[dt.date, float]],
    adjustments: Sequence[Adjustment] = (),
) -> Forecast:
    """Build the 90-day balance path under a given set of spending changes."""
    horizon = as_of + dt.timedelta(days=HORIZON)
    stopped = {a.event_id for a in adjustments if a.kind == "stop"}
    reduced = {a.event_id: a.new_amount for a in adjustments if a.kind == "reduce_to"}
    covered = {r.category for r in recurring}

    deltas: list[tuple[dt.date, float]] = list(one_offs)

    for flow in recurring:
        if flow.representative_id in stopped:
            continue
        amount = flow.amount
        if flow.representative_id in reduced and reduced[flow.representative_id] is not None:
            amount = -abs(float(reduced[flow.representative_id]))
        if 26 <= flow.period_days <= 32:
            for step in range(1, 13):
                day = add_months(flow.last_date, step)
                if day > horizon:
                    break
                if day >= as_of:
                    deltas.append((day, amount))
        else:
            day = flow.last_date
            for _ in range(400):
                day = day + dt.timedelta(days=flow.period_days)
                if day > horizon:
                    break
                if day >= as_of:
                    deltas.append((day, amount))

    for event in events:
        if as_of < event.date <= horizon and event.category not in covered:
            deltas.append((event.date, event.signed))

    deltas.sort(key=lambda d: d[0])
    return Forecast(as_of, start_balance, minimum_balance, horizon, deltas)


def candidate_adjustments(
    recurring: Sequence[Recurring], profile: Mapping[str, str]
) -> list[list[Adjustment]]:
    """Permitted spending changes: no change, each single change, then all of them."""
    stop_ok = set(split_list(profile.get("expense_categories_user_is_willing_to_stop", "")))
    reduce_ok = set(split_list(profile.get("expense_categories_user_is_willing_to_reduce", "")))

    singles: list[Adjustment] = []
    for flow in recurring:
        if flow.amount >= 0:
            continue
        if flow.stoppable and flow.category in stop_ok:
            singles.append(Adjustment(flow.representative_id, "stop"))
        elif flow.reducible and flow.category in reduce_ok:
            floor = flow.minimum_allowed if flow.minimum_allowed is not None else abs(flow.amount) / 2
            singles.append(Adjustment(flow.representative_id, "reduce_to", floor))

    sets: list[list[Adjustment]] = [[]]
    sets += [[a] for a in singles]
    if len(singles) > 1:
        sets.append(singles[:3])
    return sets


# --------------------------------------------------------------------------- #
# Per-request
# --------------------------------------------------------------------------- #


def solve_request(row: Mapping[str, str], dataset: Dataset) -> dict[str, Any]:
    profile = dataset.profiles.get(row["user_id"], {})
    home = profile.get("home_currency", "USD")
    as_of = parse_date(row["request_date"]) or dt.date.today()
    requested = parse_amount(row.get("requested_amount", "")) or 0.0
    deadline = parse_date(row.get("desired_completion_date", ""))
    allows_partial = (row.get("allows_partial_payment", "") or "").strip().lower() == "true"
    accepted = set(split_list(profile.get("payment_methods_user_will_consider", "")))
    # AGENTS.md 6.1: a blank max_installment_months means the user will not
    # consider installments at all, regardless of what options the seller lists.
    max_months = parse_amount(profile.get("max_installment_months", ""))
    if max_months is None:
        accepted.discard("installments")
    minimum_balance = parse_amount(profile.get("minimum_balance_to_keep", "")) or 0.0
    start_balance = parse_amount(profile.get("current_available_balance", "")) or 0.0

    events = build_events(dataset.events.get(row["user_id"], []), home,
                          dataset.rates, dataset.amount_overrides)
    recurring = infer_recurring(events, as_of, dataset.description_facts)
    recurring, one_offs = apply_amendments(recurring, dataset, row["user_id"], home, as_of)

    def forecast_fn(adjustments: Sequence[Adjustment]) -> Forecast:
        return project(as_of=as_of, start_balance=start_balance,
                       minimum_balance=minimum_balance, events=events,
                       recurring=recurring, one_offs=one_offs, adjustments=adjustments)

    base = forecast_fn([])
    amount_safe = max_safe_today(base, requested)
    earliest = earliest_full_payment_date(base, requested, as_of)

    adjustment_sets = candidate_adjustments(recurring, profile)
    candidates = build_candidates(
        request_date=as_of, requested_amount=requested, deadline=deadline,
        allows_partial=allows_partial,
        options=[o for o in dataset.options.get(row["request_id"], [])
                 if o.method != "installments" or (max_months and o.count <= max_months)],
        forecast_fn=forecast_fn, adjustment_sets=adjustment_sets,
    )
    winner = select(candidates, accepted_methods=accepted, deadline=deadline)
    status, method = classify(winner, request_date=as_of, requested_amount=requested)

    # Spec: for affordable_now the earliest date is the request date; leave it
    # empty when the full amount never becomes safe inside the forecast.
    if status == "affordable_now":
        earliest_out = as_of.isoformat()
    elif earliest is not None:
        earliest_out = earliest.isoformat()
    else:
        earliest_out = ""

    changes = winner.adjustments if winner else []
    return {
        "request_id": row["request_id"],
        "amount_safe_to_pay": round(amount_safe, 2),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": winner.render_plan() if winner else "none",
        "earliest_date_for_full_payment": earliest_out,
        "spending_changes_needed": "|".join(a.render() for a in changes) if changes else "none",
        "decision_explanation": explain(
            status, winner, currency=home, minimum_balance=minimum_balance,
            requested_amount=requested, deadline=deadline, adjustments=changes,
        ),
    }


def fallback_row(row: Mapping[str, str], error: Exception) -> dict[str, Any]:
    """A legal, conservative row beats a gap in the submission."""
    return {
        "request_id": row.get("request_id", ""),
        "amount_safe_to_pay": 0,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": (
            "This request could not be fully evaluated, so no payment is recommended."
        ),
    }
