"""Plan search: enumerate every candidate, keep the safe ones, rank, explain.

No model is consulted here. Given a forecast this module is a pure function, so
the same inputs always produce the same recommendation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .sept_state import Adjustment, Forecast, Recurring, _money, parse_amount, parse_date

FORECAST_DAYS = 90

STATUS_NOW = "affordable_now"
STATUS_PLAN = "affordable_with_plan"
STATUS_LATER = "affordable_later"
STATUS_NOT = "not_affordable"

METHOD_FULL = "full_payment"
METHOD_PARTIAL = "partial_payment"
METHOD_INSTALL = "installments"
METHOD_WAIT = "wait"
METHOD_NONE = "not_recommended"


@dataclass
class PaymentOption:
    option_id: str
    method: str
    amount: float
    count: int
    first_date: dt.date | None
    frequency_days: int | None
    fee: float
    total_payable: float

    def schedule(self) -> list[tuple[dt.date, float]]:
        if self.first_date is None:
            return []
        step = self.frequency_days or 30
        return [
            (self.first_date + dt.timedelta(days=step * i), self.amount)
            for i in range(max(1, self.count))
        ]

    @property
    def sort_key(self) -> tuple:
        # payment_option_07 -> 7, so the final tie-breaker orders numerically.
        digits = "".join(ch for ch in self.option_id if ch.isdigit())
        return (int(digits) if digits else 10**9, self.option_id)


def load_options(rows: Sequence[Mapping[str, str]]) -> list[PaymentOption]:
    out: list[PaymentOption] = []
    for row in rows:
        out.append(
            PaymentOption(
                option_id=row.get("payment_option_id", ""),
                method=(row.get("payment_method") or "").strip(),
                amount=parse_amount(row.get("payment_amount", "")) or 0.0,
                count=int(parse_amount(row.get("number_of_payments", "")) or 1),
                first_date=parse_date(row.get("first_payment_date", "")),
                frequency_days=int(parse_amount(row.get("payment_frequency_days", "")) or 0) or None,
                fee=parse_amount(row.get("financing_fee", "")) or 0.0,
                total_payable=parse_amount(row.get("total_payable_amount", "")) or 0.0,
            )
        )
    return sorted(out, key=lambda o: o.sort_key)


@dataclass
class Candidate:
    method: str
    payments: list[tuple[dt.date, float]]
    total_paid: float
    completes_on: dt.date | None
    adjustments: list[Adjustment] = field(default_factory=list)
    option: PaymentOption | None = None

    @property
    def starts_on(self) -> dt.date | None:
        return self.payments[0][0] if self.payments else None

    def render_plan(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(f"{d.isoformat()}:{_money(a)}" for d, a in self.payments)

    def ranking_key(self, deadline: dt.date | None) -> tuple:
        """The spec's ranking, lowest-is-best on every component."""
        completes = 0 if (deadline and self.completes_on and self.completes_on <= deadline) else 1
        return (
            completes,                                   # 1. complete by the deadline
            1 if self.adjustments else 0,                # 2. no spending changes
            round(self.total_paid, 2),                   # 3. minimise total paid
            self.starts_on or dt.date.max,               # 4. start earlier
            len(self.payments),                          # 5. fewer payments
            self.option.sort_key if self.option else (10**9, ""),  # 6. lowest option id
        )


ForecastFn = Callable[[Sequence[Adjustment]], Forecast]


def max_safe_today(forecast: Forecast, cap: float) -> float:
    """Largest amount payable on the forecast start date that holds the invariant.

    A payment on day zero lowers every later point by the same amount, so the
    headroom is exactly the forecast minimum less the floor - no search needed.
    """
    baseline = min(forecast.min_balance(), forecast.start_balance)
    return max(0.0, min(cap, baseline - forecast.minimum_balance))


def earliest_full_payment_date(
    forecast: Forecast, amount: float, start: dt.date, horizon_days: int = FORECAST_DAYS
) -> dt.date | None:
    """First date on which paying `amount` in one go keeps the invariant."""
    for offset in range(horizon_days + 1):
        day = start + dt.timedelta(days=offset)
        if forecast.is_safe([(day, amount)]):
            return day
    return None


def build_candidates(
    *,
    request_date: dt.date,
    requested_amount: float,
    deadline: dt.date | None,
    allows_partial: bool,
    options: Sequence[PaymentOption],
    forecast_fn: ForecastFn,
    adjustment_sets: Sequence[Sequence[Adjustment]],
) -> list[Candidate]:
    """Every plan worth considering, already filtered to the safe ones.

    The deadline is a *hard* constraint, not a preference: the spec defines a
    safe recommendation as one that completes the full request by its deadline.
    A plan finishing after `desired_completion_date` is therefore not a worse
    plan, it is not a plan at all.
    """
    safe: list[Candidate] = []

    def completes_in_time(on: dt.date | None) -> bool:
        return deadline is None or (on is not None and on <= deadline)

    for adjustments in adjustment_sets:
        forecast = forecast_fn(adjustments)
        adj = list(adjustments)

        # -- full payment today ------------------------------------------------
        full_today = [(request_date, requested_amount)]
        if forecast.is_safe(full_today) and completes_in_time(request_date):
            safe.append(
                Candidate(METHOD_FULL, full_today, requested_amount, request_date, adj,
                          _match_option(options, METHOD_FULL, requested_amount, 1))
            )

        # -- wait: full payment on the earliest safe later date ----------------
        earliest = earliest_full_payment_date(forecast, requested_amount, request_date)
        if earliest and earliest > request_date and completes_in_time(earliest):
            safe.append(
                Candidate(METHOD_WAIT, [(earliest, requested_amount)], requested_amount,
                          earliest, adj, None)
            )

        # -- partial payment: pay what is safe now, the rest when it is safe ---
        if allows_partial:
            today = max_safe_today(forecast, requested_amount)
            remainder = requested_amount - today
            if today > 0 and remainder > 0 and earliest and earliest <= (deadline or earliest):
                plan = [(request_date, today), (earliest, remainder)]
                if forecast.is_safe(plan):
                    safe.append(
                        Candidate(METHOD_PARTIAL, plan, requested_amount, earliest, adj, None)
                    )

        # -- installments: only exactly as supplied ----------------------------
        for option in options:
            if option.method != METHOD_INSTALL:
                continue
            schedule = option.schedule()
            if not schedule or not forecast.is_safe(schedule):
                continue
            if not completes_in_time(schedule[-1][0]):
                continue
            safe.append(
                Candidate(METHOD_INSTALL, schedule,
                          option.total_payable or option.amount * option.count,
                          schedule[-1][0], adj, option)
            )

    return safe


def _match_option(
    options: Sequence[PaymentOption], method: str, amount: float, count: int
) -> PaymentOption | None:
    for option in options:
        if option.method == method and abs(option.amount - amount) < 0.01 and option.count == count:
            return option
    return None


def select(
    candidates: Sequence[Candidate],
    *,
    accepted_methods: set[str],
    deadline: dt.date | None,
) -> Candidate | None:
    """Apply eligibility, then the ranking. `wait` needs full_payment accepted."""
    eligible: list[Candidate] = []
    for cand in candidates:
        needed = METHOD_FULL if cand.method == METHOD_WAIT else cand.method
        if needed in accepted_methods:
            eligible.append(cand)
    if not eligible:
        return None
    return min(eligible, key=lambda c: c.ranking_key(deadline))


def classify(
    winner: Candidate | None,
    *,
    request_date: dt.date,
    requested_amount: float,
) -> tuple[str, str]:
    """Map the winning plan to (affordability_status, recommended_payment_method)."""
    if winner is None:
        return STATUS_NOT, METHOD_NONE
    if (winner.method == METHOD_FULL
            and winner.starts_on == request_date
            and not winner.adjustments):
        return STATUS_NOW, METHOD_FULL
    if winner.method == METHOD_FULL:
        # Full payment, but only reachable via permitted spending changes.
        return STATUS_PLAN, METHOD_FULL
    if winner.method == METHOD_WAIT:
        return STATUS_LATER, METHOD_WAIT
    return STATUS_PLAN, winner.method


# --------------------------------------------------------------------------- #
# Explanations: the golden set is templated, so these are too.
# --------------------------------------------------------------------------- #


def _fmt(currency: str, amount: float) -> str:
    return f"{currency} {amount:,.2f}".replace(".00", "")


def _long_date(day: dt.date) -> str:
    return f"{day.day} {day:%B %Y}"


def explain(
    status: str,
    winner: Candidate | None,
    *,
    currency: str,
    minimum_balance: float,
    requested_amount: float,
    deadline: dt.date | None,
    adjustments: Sequence[Adjustment] = (),
) -> str:
    floor = _fmt(currency, minimum_balance)
    prefix = ""
    if adjustments:
        verb = "Stop" if adjustments[0].kind == "stop" else "Reduce"
        prefix = f"{verb} the flexible expense, then "

    if status == STATUS_NOT:
        when = _long_date(deadline) if deadline else "the requested date"
        return (
            f"Do not make this payment by {when}. None of the available options "
            f"keeps the {floor} minimum protected."
        )
    if winner is None:
        return f"No safe plan keeps the {floor} minimum protected."

    if status == STATUS_NOW:
        amount = _fmt(currency, winner.payments[0][1])
        return (
            f"{prefix}Pay {amount} today." if prefix else f"Pay {amount} today."
        ) + f" This leaves at least {floor} available over the next 90 days."

    if winner.method == METHOD_WAIT:
        day = _long_date(winner.payments[0][0])
        amount = _fmt(currency, requested_amount)
        return (
            f"Wait until {day}, then pay {amount} in full. "
            f"Paying sooner would put the {floor} minimum at risk."
        )

    if winner.method == METHOD_INSTALL:
        count = len(winner.payments)
        amount = _fmt(currency, winner.payments[0][1])
        start = _long_date(winner.payments[0][0])
        return (
            f"{prefix}Use {count} installments of {amount}, starting {start}. "
            f"This leaves at least {floor} available."
        )

    if winner.method == METHOD_PARTIAL:
        today = _fmt(currency, winner.payments[0][1])
        rest = _fmt(currency, winner.payments[1][1])
        day = _long_date(winner.payments[1][0])
        return (
            f"{prefix}Pay {today} today and the remaining {rest} on {day}. "
            f"This leaves at least {floor} available."
        )

    amount = _fmt(currency, winner.payments[0][1])
    return f"{prefix}Pay {amount} today. This leaves at least {floor} available."
