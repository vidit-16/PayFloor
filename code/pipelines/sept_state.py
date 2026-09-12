"""Financial state reconstruction: events -> a 90-day balance forecast.

This is the deterministic core. Nothing here calls a model.

The spec defines safety as an invariant over a forecast ("the balance never
falls below `minimum_balance_to_keep` for 90 days"), which makes the correct
answer *computable* rather than predictable. So the job is to reconstruct the
user's cash flow faithfully, and everything downstream is search over it.

Three things drive accuracy:

1. **Which events are real.** Status and direction decide this, and the spec is
   explicit: ignore pending *credits*, failed, cancelled, and unrealized. A
   pending *debit* is money likely to leave, so it counts — that is also the
   "financially safer interpretation" the spec asks for when in doubt.
2. **Which events recur.** There is no recurrence field. It has to be inferred
   from repeated (description, category) groups at a stable cadence.
3. **Currency.** Balances and outputs are in the user's home currency; events
   may be in others, converted at the dated rate.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

FORECAST_DAYS = 90

# Statuses that never contribute to the forecast.
DEAD_STATUSES = {"cancelled", "failed", "unrealized"}
# A non-cash line (investment valuation) never moves the bank balance.
DEAD_DIRECTIONS = {"non_cash"}


def parse_date(value: str) -> dt.date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_amount(value: str) -> float | None:
    value = (value or "").strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def split_list(value: str, sep: str = "|") -> list[str]:
    return [p.strip() for p in (value or "").split(sep) if p.strip()]


# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #


class RateTable:
    """Dated FX lookup. Falls back to the nearest earlier rate, then any rate."""

    def __init__(self, rows: Iterable[Mapping[str, str]]) -> None:
        self._by_pair: dict[tuple[str, str], list[tuple[dt.date, float]]] = defaultdict(list)
        for row in rows:
            date = parse_date(row.get("rate_date", ""))
            rate = parse_amount(row.get("rate", ""))
            if date is None or rate is None:
                continue
            self._by_pair[(row["from_currency"], row["to_currency"])].append((date, rate))
        for series in self._by_pair.values():
            series.sort()

    def _direct(self, src: str, dst: str, on: dt.date) -> float | None:
        series = self._by_pair.get((src, dst))
        if not series:
            return None
        best = None
        for date, rate in series:
            if date <= on:
                best = rate
            else:
                break
        return best if best is not None else series[0][1]

    def convert(self, amount: float, src: str, dst: str, on: dt.date) -> float:
        if not src or not dst or src == dst:
            return amount
        direct = self._direct(src, dst, on)
        if direct is not None:
            return amount * direct
        inverse = self._direct(dst, src, on)
        if inverse:
            return amount / inverse
        # Triangulate through the majors present in the table (USD, EUR).
        for pivot in ("USD", "EUR"):
            first = self._direct(src, pivot, on)
            second = self._direct(pivot, dst, on)
            if first and second:
                return amount * first * second
            inv_first = self._direct(pivot, src, on)
            if inv_first and second:
                return amount / inv_first * second
        return amount


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    """One cash movement, normalised into home currency and signed."""

    event_id: str
    date: dt.date
    signed: float            # + credit, - debit, in home currency
    category: str
    description: str
    event_type: str
    status: str
    flexibility: str
    minimum_allowed: float | None
    currency: str
    raw_amount: float | None

    @property
    def is_income(self) -> bool:
        return self.signed > 0

    @property
    def stoppable(self) -> bool:
        return self.flexibility in {"stoppable", "reducible_or_stoppable"}

    @property
    def reducible(self) -> bool:
        return self.flexibility in {"reducible", "reducible_or_stoppable"}


def counts_toward_forecast(row: Mapping[str, str]) -> bool:
    """Spec: ignore pending credits, failed or cancelled transactions,
    duplicate records, and unrealized investments.

    Cancelled/failed/unrealized are dropped outright. `non_cash` never touches
    the balance. A *pending credit* is money not yet guaranteed, so it is
    dropped; a *pending debit* is money likely to leave, so it is kept - the
    conservative reading, which is what the spec asks for on ambiguity.
    """
    status = (row.get("status") or "").strip().lower()
    direction = (row.get("direction") or "").strip().lower()
    if status in DEAD_STATUSES or direction in DEAD_DIRECTIONS:
        return False
    if status == "pending" and direction == "credit":
        return False
    return True


def build_events(
    rows: Sequence[Mapping[str, str]],
    home_currency: str,
    rates: RateTable,
    amount_overrides: Mapping[str, float] | None = None,
) -> list[Event]:
    """Normalise raw event rows into signed, home-currency `Event`s.

    `amount_overrides` carries amounts recovered from images for the events
    whose `amount` column is blank - the one place model output enters the
    deterministic core, and it enters as a number, not a decision.
    """
    overrides = amount_overrides or {}
    events: list[Event] = []
    for row in rows:
        if not counts_toward_forecast(row):
            continue
        date = parse_date(row.get("event_date", ""))
        if date is None:
            continue
        amount = parse_amount(row.get("amount", ""))
        if amount is None:
            amount = overrides.get(row.get("event_id", ""))
        if amount is None:
            continue  # blank and no image amount recovered yet
        currency = (row.get("currency") or home_currency).strip()
        # AGENTS.md 6.1: convert at the rate for the SETTLEMENT date, which
        # is when the cash actually moves, falling back to the event date.
        fx_date = parse_date(row.get("settlement_date", "")) or date
        converted = rates.convert(amount, currency, home_currency, fx_date)
        direction = (row.get("direction") or "").strip().lower()
        signed = converted if direction == "credit" else -converted
        events.append(
            Event(
                event_id=row.get("event_id", ""),
                date=date,
                signed=signed,
                category=(row.get("category") or "").strip(),
                description=(row.get("description") or "").strip(),
                event_type=(row.get("event_type") or "").strip(),
                status=(row.get("status") or "").strip(),
                flexibility=(row.get("flexibility") or "").strip(),
                minimum_allowed=parse_amount(row.get("minimum_allowed_amount", "")),
                currency=currency,
                raw_amount=amount,
            )
        )
    events.sort(key=lambda e: (e.date, e.event_id))
    return events


# --------------------------------------------------------------------------- #
# Recurrence
# --------------------------------------------------------------------------- #


@dataclass
class Recurring:
    """An inferred repeating cash flow, projected forward over the forecast."""

    key: tuple[str, str]
    period_days: int
    amount: float            # signed, home currency
    last_date: dt.date
    category: str
    description: str
    flexibility: str
    minimum_allowed: float | None
    event_ids: list[str] = field(default_factory=list)
    representative_id: str = ""

    @property
    def stoppable(self) -> bool:
        return self.flexibility in {"stoppable", "reducible_or_stoppable"}

    @property
    def reducible(self) -> bool:
        return self.flexibility in {"reducible", "reducible_or_stoppable"}

    def occurrences(self, start: dt.date, end: dt.date) -> list[dt.date]:
        """Future dates this flow is expected to land on, within (start, end]."""
        dates: list[dt.date] = []
        cursor = self.last_date
        guard = 0
        while guard < 400:
            guard += 1
            cursor = cursor + dt.timedelta(days=self.period_days)
            if cursor > end:
                break
            if cursor > start:
                dates.append(cursor)
        return dates


#: Cadences we treat as genuine recurrence, with the tolerance used to match.
_CADENCES = ((30, 6), (14, 3), (7, 2))


def detect_recurring(events: Sequence[Event], as_of: dt.date) -> list[Recurring]:
    """Infer repeating flows from history.

    Grouped by (description, category) because that is what actually repeats in
    this data: 'Apartment rent transfer' lands every 30-31 days, while
    'Neighbourhood grocer' lands at irregular intervals and is variable
    spending rather than a commitment. Requiring a stable cadence over at least
    three observations keeps the latter out.
    """
    groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for event in events:
        if event.date <= as_of:
            groups[(event.description, event.category)].append(event)

    recurring: list[Recurring] = []
    for key, group in groups.items():
        if len(group) < 3:
            continue
        group.sort(key=lambda e: e.date)
        gaps = [
            (group[i + 1].date - group[i].date).days for i in range(len(group) - 1)
        ]
        if not gaps:
            continue
        median_gap = statistics.median(gaps)
        matched = None
        for period, tolerance in _CADENCES:
            if abs(median_gap - period) <= tolerance:
                # Require most gaps to sit near that cadence, not just the median.
                close = sum(1 for g in gaps if abs(g - period) <= tolerance)
                if close >= max(2, len(gaps) - 1):
                    matched = period
                    break
        if matched is None:
            continue
        amounts = [e.signed for e in group]
        recurring.append(
            Recurring(
                key=key,
                period_days=matched,
                amount=statistics.median(amounts),
                last_date=group[-1].date,
                category=group[-1].category,
                description=group[-1].description,
                flexibility=group[-1].flexibility,
                minimum_allowed=group[-1].minimum_allowed,
                event_ids=[e.event_id for e in group],
                representative_id=group[-1].event_id,
            )
        )
    return recurring


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #


@dataclass
class Adjustment:
    """A spending change applied to the forecast (stop or reduce a flow)."""

    event_id: str
    kind: str                 # "stop" | "reduce_to"
    new_amount: float | None = None

    def render(self) -> str:
        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{_money(self.new_amount or 0.0)}"


def _money(value: float) -> str:
    rounded = round(value + 1e-9, 2)
    return str(int(rounded)) if abs(rounded - int(rounded)) < 1e-9 else f"{rounded:.2f}"


@dataclass
class Forecast:
    """Everything needed to test a candidate plan against the safety invariant."""

    start_date: dt.date
    start_balance: float
    minimum_balance: float
    horizon: dt.date
    deltas: list[tuple[dt.date, float]]

    def balance_series(
        self, payments: Sequence[tuple[dt.date, float]] = ()
    ) -> list[tuple[dt.date, float]]:
        moves: list[tuple[dt.date, float]] = list(self.deltas)
        moves += [(date, -amount) for date, amount in payments]
        moves.sort(key=lambda m: m[0])
        balance = self.start_balance
        series: list[tuple[dt.date, float]] = []
        for date, delta in moves:
            balance += delta
            series.append((date, balance))
        return series

    def min_balance(self, payments: Sequence[tuple[dt.date, float]] = ()) -> float:
        series = self.balance_series(payments)
        return min([b for _, b in series], default=self.start_balance)

    def is_safe(self, payments: Sequence[tuple[dt.date, float]] = ()) -> bool:
        # A tiny epsilon so float noise never flips a boundary case.
        return self.min_balance(payments) >= self.minimum_balance - 1e-6


def build_forecast(
    *,
    as_of: dt.date,
    start_balance: float,
    minimum_balance: float,
    events: Sequence[Event],
    recurring: Sequence[Recurring],
    adjustments: Sequence[Adjustment] = (),
    horizon_days: int = FORECAST_DAYS,
) -> Forecast:
    """Project the balance forward.

    Future cash flow is the union of events already dated ahead of `as_of`
    (scheduled income, pending debits, confirmed payments) and the projected
    occurrences of each inferred recurring flow. `adjustments` suppress or
    shrink recurring flows the user has said they are willing to change.
    """
    horizon = as_of + dt.timedelta(days=horizon_days)
    stopped = {a.event_id for a in adjustments if a.kind == "stop"}
    reduced = {a.event_id: a.new_amount for a in adjustments if a.kind == "reduce_to"}

    deltas: list[tuple[dt.date, float]] = []

    # Known future-dated events.
    recurring_ids = {eid for r in recurring for eid in r.event_ids}
    for event in events:
        if as_of < event.date <= horizon and event.event_id not in recurring_ids:
            deltas.append((event.date, event.signed))

    # Projected recurring flows.
    for flow in recurring:
        if flow.representative_id in stopped:
            continue
        amount = flow.amount
        if flow.representative_id in reduced and reduced[flow.representative_id] is not None:
            # Reductions apply to spending, so keep the sign and cap magnitude.
            amount = -abs(reduced[flow.representative_id])  # type: ignore[arg-type]
        for date in flow.occurrences(as_of, horizon):
            deltas.append((date, amount))

    deltas.sort(key=lambda d: d[0])
    return Forecast(
        start_date=as_of,
        start_balance=start_balance,
        minimum_balance=minimum_balance,
        horizon=horizon,
        deltas=deltas,
    )
