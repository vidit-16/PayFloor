"""Property tests over the real dataset.

The 25 labelled samples measure *accuracy*. These measure *correctness* — the
properties that must hold for all 250 rows regardless of whether the answer
matches gold. A pipeline can score well on the samples and still ship rows that
violate the spec's own invariants on the hidden set, and only this catches that.

Run:  python tests/test_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pipelines.september2026 as P  # noqa: E402
from pipelines.sept_state import parse_amount, parse_date  # noqa: E402
from pipelines.sept_validate import parse_plan, validate_all  # noqa: E402

DATASET = P.Dataset("../dataset")
REQUESTS = DATASET.requests
PREDS = [P.solve_request(r, DATASET) for r in REQUESTS]
BY_ID = {p["request_id"]: p for p in PREDS}


def test_every_request_gets_exactly_one_prediction():
    assert len(PREDS) == len(REQUESTS)
    assert len(BY_ID) == len(REQUESTS), "duplicate request_id in predictions"


def test_amount_is_within_spec_bounds():
    """0 <= amount_safe_to_pay <= requested_amount, stated explicitly in the spec."""
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        requested = parse_amount(request["requested_amount"]) or 0.0
        safe = float(pred["amount_safe_to_pay"])
        assert -0.02 <= safe <= requested + 0.02, (
            f"{request['request_id']}: {safe} outside [0, {requested}]"
        )


def test_no_cross_field_violations():
    """Every row must be internally coherent, not merely per-column legal."""
    results = validate_all(
        REQUESTS, PREDS,
        profiles=DATASET.profiles,
        options_by_request=DATASET.options,
    )
    bad = [r for r in results if not r.ok]
    if bad:
        for r in bad[:10]:
            print(f"    {r.request_id}: {'; '.join(r.violations)}")
    assert not bad, f"{len(bad)} rows with cross-field violations"


def test_every_recommended_plan_actually_holds_the_invariant():
    """The decisive property: a plan we recommend must survive its own forecast.

    This re-simulates each shipped plan independently of the solver that chose
    it. If the planner and the simulator ever disagree, this fails.
    """
    failures = []
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        if pred["recommended_payment_method"] == "not_recommended":
            continue
        plan = parse_plan(str(pred["payment_plan"]))
        if not plan:
            continue
        profile = DATASET.profiles[request["user_id"]]
        # Re-simulate under the plan's OWN spending changes: a plan that is
        # only safe after stopping a subscription is still a valid plan, and
        # judging it against the unadjusted forecast would be wrong.
        forecast = _forecast_for(request, profile,
                                 _adjustments_of(str(pred["spending_changes_needed"])))
        # Checked against the profile's floor directly, NOT via is_safe(). This
        # test exists to catch a wrong safety check, so it cannot use the safety
        # check to do it - mutation testing showed a version that did would
        # pass even with the minimum-balance floor removed entirely.
        floor = parse_amount(profile["minimum_balance_to_keep"]) or 0.0
        lowest = forecast.min_balance(plan)
        if lowest < floor - 1e-6:
            failures.append(f"{request['request_id']} breaches by {floor - lowest:,.2f}")
    if failures:
        for f in failures[:10]:
            print(f"    {f}")
    assert not failures, f"{len(failures)} recommended plans breach the minimum balance"


def test_safe_amount_is_monotonic():
    """If X is safe to pay today, every smaller amount is too."""
    for request in REQUESTS[:60]:
        profile = DATASET.profiles[request["user_id"]]
        forecast = _forecast_for(request, profile)
        safe = float(BY_ID[request["request_id"]]["amount_safe_to_pay"])
        if safe <= 0:
            continue
        as_of = parse_date(request["request_date"])
        for fraction in (0.25, 0.5, 0.9):
            smaller = safe * fraction
            assert forecast.is_safe([(as_of, smaller)]), (
                f"{request['request_id']}: {safe} safe but {smaller} is not"
            )


def test_pipeline_is_deterministic():
    """Same input, same output — twice, on a sample of rows."""
    for request in REQUESTS[:40]:
        first = P.solve_request(request, DATASET)
        second = P.solve_request(request, DATASET)
        assert first == second, f"{request['request_id']} is not deterministic"


def test_cancelled_and_superseded_events_never_reach_the_forecast():
    """No double counting: a cancelled authorisation and its settled twin must
    not both move the balance."""
    for request in REQUESTS[:40]:
        raw = DATASET.events.get(request["user_id"], [])
        dead = {
            r["event_id"] for r in raw
            if (r.get("status") or "").lower() in {"cancelled", "failed", "unrealized"}
        }
        built = P.build_events(
            raw, DATASET.profiles[request["user_id"]]["home_currency"],
            DATASET.rates, DATASET.amount_overrides,
        )
        leaked = dead & {e.event_id for e in built}
        assert not leaked, f"{request['user_id']}: dead events in forecast: {leaked}"


def test_pending_credits_excluded_pending_debits_kept():
    """The spec's asymmetry, which is easy to get backwards."""
    for request in REQUESTS[:40]:
        raw = DATASET.events.get(request["user_id"], [])
        built = {e.event_id for e in P.build_events(
            raw, DATASET.profiles[request["user_id"]]["home_currency"],
            DATASET.rates, DATASET.amount_overrides)}
        for row in raw:
            if (row.get("status") or "").lower() != "pending":
                continue
            direction = (row.get("direction") or "").lower()
            if direction == "credit":
                assert row["event_id"] not in built, f"pending credit counted: {row['event_id']}"
            elif direction == "debit" and row.get("amount"):
                assert row["event_id"] in built, f"pending debit dropped: {row['event_id']}"


def test_no_recurrence_invented_from_a_single_discretionary_event():
    """One isolated discretionary expense must not become a recurring flow."""
    for request in REQUESTS[:40]:
        as_of = parse_date(request["request_date"])
        home = DATASET.profiles[request["user_id"]]["home_currency"]
        events = P.build_events(DATASET.events.get(request["user_id"], []),
                                home, DATASET.rates, DATASET.amount_overrides)
        recurring = P.infer_recurring(events, as_of, DATASET.description_facts)
        for flow in recurring:
            if len(flow.event_ids) == 1:
                facts = DATASET.description_facts.get(flow.description, {})
                assert facts.get("is_commitment"), (
                    f"{request['user_id']}: recurrence invented from one "
                    f"non-commitment event ({flow.description})"
                )


def test_spending_changes_only_touch_permitted_flexible_events():
    """A change must be permitted by flexibility AND by the user's stated
    willingness — never a protected category."""
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        raw_changes = str(pred["spending_changes_needed"])
        if raw_changes == "none":
            continue
        profile = DATASET.profiles[request["user_id"]]
        protected = set(profile.get("expense_categories_to_protect", "").split("|"))
        events = {e["event_id"]: e for e in DATASET.events.get(request["user_id"], [])}
        for change in raw_changes.split("|"):
            parts = change.split(":")
            event = events.get(parts[1] if len(parts) > 1 else "")
            assert event is not None, f"{request['request_id']}: unknown event {change}"
            assert event["flexibility"] != "fixed", (
                f"{request['request_id']}: changed a fixed event {change}"
            )
            assert event["category"] not in protected, (
                f"{request['request_id']}: changed a protected category {event['category']}"
            )


def test_recommended_method_is_one_the_user_accepts():
    """The user's payment_methods_user_will_consider is a hard filter, not a
    preference. Added after mutation testing showed nothing enforced it."""
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        method = pred["recommended_payment_method"]
        if method == "not_recommended":
            continue
        profile = DATASET.profiles[request["user_id"]]
        accepted = set(profile.get("payment_methods_user_will_consider", "").split("|"))
        needed = "full_payment" if method == "wait" else method
        assert needed in accepted, (
            f"{request['request_id']}: recommended {method} but the user accepts "
            f"{sorted(accepted)}"
        )


def test_installments_respect_max_installment_months():
    """A blank max_installment_months means the user rejects installments."""
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        if pred["recommended_payment_method"] != "installments":
            continue
        profile = DATASET.profiles[request["user_id"]]
        cap = parse_amount(profile.get("max_installment_months", ""))
        assert cap is not None, f"{request['request_id']}: installments with no stated cap"
        count = len(parse_plan(str(pred["payment_plan"])))
        assert count <= cap, f"{request['request_id']}: {count} payments exceeds cap {cap}"


def test_installment_plans_match_the_raw_option_rows():
    """Checked against the CSV itself, not PaymentOption.schedule().

    The validator compares a plan with option.schedule(), the same code that
    built the plan, so a bug in schedule() moves both sides together. Mutation
    testing showed exactly that: an altered installment amount changed 11 rows
    and raised no violation. This rebuilds each schedule from the raw columns.
    """
    import csv
    import datetime as dt

    with open("../dataset/request_payment_options.csv", newline="", encoding="utf-8-sig") as fh:
        raw = [row for row in csv.DictReader(fh) if row["payment_method"] == "installments"]
    schedules: dict[str, list[list[tuple]]] = {}
    for row in raw:
        first = dt.date.fromisoformat(row["first_payment_date"])
        step = int(row["payment_frequency_days"])
        schedules.setdefault(row["request_id"], []).append([
            (first + dt.timedelta(days=step * i), round(float(row["payment_amount"]), 2))
            for i in range(int(row["number_of_payments"]))
        ])
    checked = 0
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        if pred["recommended_payment_method"] != "installments":
            continue
        plan = [(day, round(amount, 2)) for day, amount in parse_plan(str(pred["payment_plan"]))]
        assert plan in schedules.get(request["request_id"], []), (
            f"{request['request_id']}: installment plan matches no supplied option row"
        )
        checked += 1
    assert checked, "no installment recommendations - this test checked nothing"


def test_spending_changes_are_in_the_users_willing_lists():
    """Flexibility alone is not permission: the category must also appear in the
    user's willing-to-stop or willing-to-reduce list. Added after mutation
    testing showed an and/or swap went undetected."""
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        raw = str(pred["spending_changes_needed"])
        if raw == "none":
            continue
        profile = DATASET.profiles[request["user_id"]]
        stop_ok = set(x for x in profile.get(
            "expense_categories_user_is_willing_to_stop", "").split("|") if x)
        reduce_ok = set(x for x in profile.get(
            "expense_categories_user_is_willing_to_reduce", "").split("|") if x)
        events = {e["event_id"]: e for e in DATASET.events.get(request["user_id"], [])}
        for change in raw.split("|"):
            parts = change.split(":")
            event = events[parts[1]]
            permitted = stop_ok if parts[0] == "stop" else reduce_ok
            assert event["category"] in permitted, (
                f"{request['request_id']}: {parts[0]} on {event['category']}, "
                f"which the user did not agree to"
            )


def test_forecast_horizon_is_ninety_days():
    """The spec fixes the window at 90 days; a shorter one hides late breaches."""
    assert P.HORIZON == 90
    request = REQUESTS[0]
    profile = DATASET.profiles[request["user_id"]]
    forecast = _forecast_for(request, profile)
    span = (forecast.horizon - forecast.start_date).days
    assert span == 90, f"forecast spans {span} days, not 90"


def test_safe_amount_is_never_rounded_above_what_is_safe():
    """amount_safe_to_pay must itself survive the forecast, to the cent."""
    for request in REQUESTS[:80]:
        pred = BY_ID[request["request_id"]]
        safe = float(pred["amount_safe_to_pay"])
        if safe <= 0:
            continue
        profile = DATASET.profiles[request["user_id"]]
        forecast = _forecast_for(request, profile)
        as_of = parse_date(request["request_date"])
        assert forecast.is_safe([(as_of, safe)]), (
            f"{request['request_id']}: amount_safe_to_pay {safe} is not itself safe"
        )


def test_partial_payment_never_uses_spending_changes():
    """Regression: the spec pins a partial plan to amount_safe_to_pay and
    earliest_date_for_full_payment, both defined before spending changes, so a
    partial plan built under spending changes contradicts its own row. The
    competition data never triggered this; synthetic data did."""
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        if pred["recommended_payment_method"] == "partial_payment":
            assert pred["spending_changes_needed"] == "none", (
                f"{request['request_id']}: partial payment combined with spending changes"
            )


def test_spending_changes_only_when_no_undisrupted_plan_exists():
    """Ranking rule 2: prefer a plan that needs no spending changes.

    If a row recommends stopping or reducing an expense, there must be no
    eligible, safe, on-time plan that leaves the user's spending alone.
    Mutation testing showed inverting this rule changed 112 of 250 rows and
    nothing noticed. Checked by rebuilding the candidates with no adjustments,
    independently of the ranking function under test.
    """
    from pipelines.sept_solver import build_candidates
    from pipelines.sept_state import split_list

    offenders = []
    for request in REQUESTS:
        pred = BY_ID[request["request_id"]]
        if str(pred["spending_changes_needed"]) == "none":
            continue
        profile = DATASET.profiles[request["user_id"]]
        accepted = set(split_list(profile.get("payment_methods_user_will_consider", "")))
        if parse_amount(profile.get("max_installment_months", "")) is None:
            accepted.discard("installments")
        forecast = _forecast_for(request, profile)
        cap = parse_amount(profile.get("max_installment_months", ""))
        options = [o for o in DATASET.options.get(request["request_id"], [])
                   if o.method != "installments" or (cap and o.count <= cap)]
        undisrupted = build_candidates(
            request_date=parse_date(request["request_date"]),
            requested_amount=parse_amount(request["requested_amount"]) or 0.0,
            deadline=parse_date(request["desired_completion_date"]),
            allows_partial=(request.get("allows_partial_payment", "").lower() == "true"),
            options=options, forecast_fn=lambda _adj: forecast, adjustment_sets=[[]],
        )
        eligible = [c for c in undisrupted
                    if ("full_payment" if c.method == "wait" else c.method) in accepted]
        if eligible:
            offenders.append(f"{request['request_id']}: changes spending although "
                             f"{eligible[0].method} needs none")
    for o in offenders[:10]:
        print(f"    {o}")
    assert not offenders, f"{len(offenders)} rows break ranking rule 2"


def _adjustments_of(text: str) -> list:
    """Parse a spending_changes_needed cell back into Adjustment objects."""
    from pipelines.sept_state import Adjustment
    out = []
    if not text or text == "none":
        return out
    for change in text.split("|"):
        parts = change.split(":")
        if parts[0] == "stop" and len(parts) == 2:
            out.append(Adjustment(parts[1], "stop"))
        elif parts[0] == "reduce_to" and len(parts) == 3:
            out.append(Adjustment(parts[1], "reduce_to", float(parts[2])))
    return out


def _forecast_for(request, profile, adjustments=()):
    as_of = parse_date(request["request_date"])
    home = profile["home_currency"]
    events = P.build_events(DATASET.events.get(request["user_id"], []),
                            home, DATASET.rates, DATASET.amount_overrides)
    recurring = P.infer_recurring(events, as_of, DATASET.description_facts)
    recurring, one_offs = P.apply_amendments(recurring, DATASET, request["user_id"], home, as_of)
    return P.project(
        as_of=as_of,
        start_balance=parse_amount(profile["current_available_balance"]) or 0.0,
        minimum_balance=parse_amount(profile["minimum_balance_to_keep"]) or 0.0,
        events=events, recurring=recurring, one_offs=one_offs,
        adjustments=adjustments,
    )


if __name__ == "__main__":
    print(f"running invariants over all {len(REQUESTS)} requests\n")
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all invariants hold' if not failures else f'{failures} invariant failure(s)'}")
    raise SystemExit(1 if failures else 0)
