"""Cross-field validation of a decision row.

`OutputSpec` checks each column in isolation: is the label in the vocabulary, is
the number in range, is the date well-formed. That cannot catch a row where
every cell is individually legal but the row as a whole is incoherent — a
`partial_payment` whose two payments do not sum to the requested amount, or an
`affordable_now` whose earliest date is not the request date.

The spec states those relationships explicitly, so they are checkable. Each is a
hard requirement, and a violation means a row that would score zero on several
columns at once while looking fine to a per-column validator.

Returns reasons rather than a bare boolean so a failure says what broke.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .sept_state import parse_amount, parse_date, split_list

TOLERANCE = 0.02  # currency rounding slack, in units of the home currency


@dataclass
class DecisionValidation:
    request_id: str
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, code: str, detail: str = "") -> None:
        self.violations.append(f"{code}{': ' + detail if detail else ''}")


def parse_plan(plan: str) -> list[tuple[dt.date, float]]:
    """`YYYY-MM-DD:amount|...` -> pairs. Raises nothing; bad parts are dropped."""
    if not plan or plan.strip().lower() == "none":
        return []
    out: list[tuple[dt.date, float]] = []
    for part in plan.split("|"):
        piece = part.strip()
        if not piece or ":" not in piece:
            continue
        date_text, _, amount_text = piece.rpartition(":")
        day = parse_date(date_text)
        amount = parse_amount(amount_text)
        if day is not None and amount is not None:
            out.append((day, amount))
    return out


def validate_decision(
    request: Mapping[str, str],
    pred: Mapping[str, Any],
    *,
    profile: Mapping[str, str] | None = None,
    options: Sequence[Any] = (),
) -> DecisionValidation:
    """Check every cross-field relationship the specification states."""
    result = DecisionValidation(request_id=str(pred.get("request_id", "")))

    requested = parse_amount(request.get("requested_amount", "")) or 0.0
    request_date = parse_date(request.get("request_date", ""))
    deadline = parse_date(request.get("desired_completion_date", ""))
    allows_partial = (request.get("allows_partial_payment", "") or "").strip().lower() == "true"

    status = str(pred.get("affordability_status", ""))
    method = str(pred.get("recommended_payment_method", ""))
    safe = parse_amount(str(pred.get("amount_safe_to_pay", ""))) or 0.0
    earliest = parse_date(str(pred.get("earliest_date_for_full_payment", "")))
    plan = parse_plan(str(pred.get("payment_plan", "")))
    changes = [c for c in split_list(str(pred.get("spending_changes_needed", "none"))) if c != "none"]

    # -- the spec's explicit numeric bound ----------------------------------
    if not (-TOLERANCE <= safe <= requested + TOLERANCE):
        result.add("AMOUNT_OUT_OF_BOUNDS", f"{safe} not in [0, {requested}]")

    # -- status / method must agree ----------------------------------------
    allowed_methods = {
        "affordable_now": {"full_payment"},
        "affordable_later": {"wait"},
        "not_affordable": {"not_recommended"},
        "affordable_with_plan": {"full_payment", "partial_payment", "installments"},
    }
    if status in allowed_methods and method not in allowed_methods[status]:
        result.add("STATUS_METHOD_MISMATCH", f"{status} with {method}")

    # -- earliest date semantics -------------------------------------------
    if status == "affordable_now" and earliest != request_date:
        result.add("EARLIEST_NOT_REQUEST_DATE", f"{earliest} != {request_date}")
    if status == "not_affordable" and method == "not_recommended" and plan:
        result.add("PLAN_ON_NOT_RECOMMENDED", "not_recommended must carry no payments")

    # -- plan shape ---------------------------------------------------------
    if method == "not_recommended":
        if plan:
            result.add("PLAN_SHOULD_BE_NONE")
    else:
        if not plan:
            result.add("PLAN_MISSING", f"{method} requires payments")
        if plan != sorted(plan, key=lambda p: p[0]):
            result.add("PLAN_NOT_CHRONOLOGICAL")
        if any(amount <= 0 for _, amount in plan):
            result.add("PLAN_NONPOSITIVE_PAYMENT")
        if deadline and plan and plan[-1][0] > deadline:
            result.add("PLAN_COMPLETES_AFTER_DEADLINE", f"{plan[-1][0]} > {deadline}")
        if request_date and plan and plan[0][0] < request_date:
            result.add("PLAN_PAYS_BEFORE_REQUEST", f"{plan[0][0]} < {request_date}")

    # -- partial payment has an exact shape in the spec ---------------------
    if method == "partial_payment":
        if status != "affordable_with_plan":
            result.add("PARTIAL_WRONG_STATUS", status)
        if not allows_partial:
            result.add("PARTIAL_NOT_PERMITTED")
        if len(plan) != 2:
            result.add("PARTIAL_NOT_TWO_PAYMENTS", str(len(plan)))
        else:
            total = plan[0][1] + plan[1][1]
            if abs(total - requested) > TOLERANCE:
                result.add("PARTIAL_SUM_MISMATCH", f"{total} != {requested}")
            if request_date and plan[0][0] != request_date:
                result.add("PARTIAL_FIRST_NOT_TODAY")
            if abs(plan[0][1] - safe) > TOLERANCE:
                result.add("PARTIAL_FIRST_NOT_SAFE_AMOUNT")
            if earliest and plan[1][0] != earliest:
                result.add("PARTIAL_SECOND_NOT_EARLIEST")
        if not (0 < safe < requested):
            result.add("PARTIAL_AMOUNT_NOT_STRICTLY_BETWEEN")

    # -- full payment covers the whole request ------------------------------
    if method == "full_payment" and plan:
        total = sum(amount for _, amount in plan)
        if abs(total - requested) > TOLERANCE:
            result.add("FULL_SUM_MISMATCH", f"{total} != {requested}")

    # -- installments must match a supplied option exactly ------------------
    if method == "installments":
        matched = False
        for option in options:
            schedule = option.schedule()
            if len(schedule) == len(plan) and all(
                d1 == d2 and abs(a1 - a2) <= TOLERANCE
                for (d1, a1), (d2, a2) in zip(schedule, plan)
            ):
                matched = True
                break
        if not matched:
            result.add("INSTALLMENTS_NOT_A_SUPPLIED_OPTION")

    # -- spending changes must be permitted and well-formed -----------------
    if len(changes) > 3:
        result.add("TOO_MANY_SPENDING_CHANGES", str(len(changes)))
    touched: set[str] = set()
    for change in changes:
        parts = change.split(":")
        if parts[0] == "stop" and len(parts) == 2:
            event_id = parts[1]
        elif parts[0] == "reduce_to" and len(parts) == 3:
            event_id = parts[1]
            if parse_amount(parts[2]) is None:
                result.add("REDUCE_AMOUNT_NOT_NUMERIC", change)
        else:
            result.add("SPENDING_CHANGE_MALFORMED", change)
            continue
        if event_id in touched:
            result.add("SPENDING_CHANGE_DUPLICATE_EVENT", event_id)
        touched.add(event_id)

    # -- explanation must exist and be grounded -----------------------------
    explanation = str(pred.get("decision_explanation", "")).strip()
    if len(explanation.split()) < 6:
        result.add("EXPLANATION_TOO_SHORT")

    return result


def validate_all(
    requests: Sequence[Mapping[str, str]],
    preds: Sequence[Mapping[str, Any]],
    *,
    profiles: Mapping[str, Mapping[str, str]] | None = None,
    options_by_request: Mapping[str, Sequence[Any]] | None = None,
) -> list[DecisionValidation]:
    by_id = {str(p.get("request_id")): p for p in preds}
    out: list[DecisionValidation] = []
    for request in requests:
        pred = by_id.get(request["request_id"])
        if pred is None:
            missing = DecisionValidation(request_id=request["request_id"])
            missing.add("PREDICTION_MISSING")
            out.append(missing)
            continue
        out.append(
            validate_decision(
                request, pred,
                profile=(profiles or {}).get(request.get("user_id", "")),
                options=(options_by_request or {}).get(request["request_id"], ()),
            )
        )
    return out
