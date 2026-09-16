"""Unit tests for the output contract, the cache, and cross-field validation.

These are fast and need no dataset. `tests/test_invariants.py` covers the
properties that require the real data.

Run:  python tests/test_harness.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from orchestrate.cache import CallCache, content_key  # noqa: E402
from pipelines.sept_state import RateTable, build_events  # noqa: E402
from pipelines.sept_validate import parse_plan, validate_decision  # noqa: E402
from pipelines.september2026 import SPEC  # noqa: E402

ROW = {
    "request_id": "r1",
    "amount_safe_to_pay": 500,
    "affordability_status": "affordable_now",
    "recommended_payment_method": "full_payment",
    "payment_plan": "2026-01-03:500",
    "earliest_date_for_full_payment": "2026-01-03",
    "spending_changes_needed": "none",
    "decision_explanation": "Pay EUR 500 today. This leaves at least EUR 100 available.",
}
REQUEST = {
    "request_id": "r1", "user_id": "u1", "request_date": "2026-01-03",
    "requested_amount": "500", "desired_completion_date": "2026-01-20",
    "allows_partial_payment": "true",
}


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #

def test_categorical_snaps_to_nearest_legal_label():
    col = SPEC.column("affordability_status")
    assert col.coerce("affordable_now") == "affordable_now"
    assert col.coerce("AFFORDABLE_NOW") == "affordable_now"
    assert col.coerce("affordable_nw") == "affordable_now"      # typo repaired
    assert col.coerce("something else") == "not_affordable"     # -> fallback


def test_large_amounts_never_use_scientific_notation():
    """%g flips to 1.5656e+07 above a million, which would corrupt every
    IDR and INR amount in the file."""
    col = SPEC.column("amount_safe_to_pay")
    assert col.coerce(15656000) == "15656000"
    assert col.coerce(60496000.5) == "60496000.5"
    assert col.coerce(1.5656e7) == "15656000"
    assert "e" not in col.coerce(4.883e7).lower()


def test_amount_coercion_handles_junk():
    col = SPEC.column("amount_safe_to_pay")
    assert col.coerce("not a number") == "0"
    assert col.coerce("") == "0"
    assert col.coerce("1,234.50") == "1234.5"


def test_empty_date_is_legal_but_malformed_date_is_not():
    """The spec requires a blank date when no full payment is ever safe."""
    col = SPEC.column("earliest_date_for_full_payment")
    assert col.coerce("") == ""
    assert col.issues("") == []
    assert col.issues("2026-01-03") == []
    assert col.issues("03/01/2026") != []


def test_validator_catches_missing_and_duplicate_rows():
    rows = [SPEC.coerce_row(ROW), SPEC.coerce_row(ROW)]
    report = SPEC.validate(rows, ["r1", "r2"])
    assert not report.ok
    assert report.missing_keys == ["r2"]
    assert report.duplicate_keys == ["r1"]


def test_validator_passes_a_clean_frame():
    assert SPEC.validate([SPEC.coerce_row(ROW)], ["r1"]).ok


def test_write_csv_emits_exact_header_order():
    target = Path(tempfile.mkdtemp()) / "out.csv"
    SPEC.write_csv([ROW], target)
    header = target.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "request_id,amount_safe_to_pay,affordability_status,"
        "recommended_payment_method,payment_plan,earliest_date_for_full_payment,"
        "spending_changes_needed,decision_explanation"
    )


# --------------------------------------------------------------------------- #
# Cross-field validation
# --------------------------------------------------------------------------- #

def test_clean_row_has_no_violations():
    assert validate_decision(REQUEST, ROW).ok


def test_status_and_method_must_agree():
    bad = {**ROW, "recommended_payment_method": "installments"}
    assert any("STATUS_METHOD_MISMATCH" in v for v in validate_decision(REQUEST, bad).violations)


def test_affordable_now_requires_earliest_equal_to_request_date():
    bad = {**ROW, "earliest_date_for_full_payment": "2026-02-01"}
    assert any("EARLIEST_NOT_REQUEST_DATE" in v for v in validate_decision(REQUEST, bad).violations)


def test_partial_payment_must_have_two_payments_summing_to_the_request():
    bad = {
        **ROW,
        "affordability_status": "affordable_with_plan",
        "recommended_payment_method": "partial_payment",
        "amount_safe_to_pay": 200,
        "payment_plan": "2026-01-03:200|2026-01-10:250",   # sums to 450, not 500
        "earliest_date_for_full_payment": "2026-01-10",
    }
    violations = validate_decision(REQUEST, bad).violations
    assert any("PARTIAL_SUM_MISMATCH" in v for v in violations)


def test_plan_completing_after_the_deadline_is_rejected():
    bad = {**ROW, "payment_plan": "2026-03-01:500",
           "affordability_status": "affordable_later",
           "recommended_payment_method": "wait",
           "earliest_date_for_full_payment": "2026-03-01"}
    assert any("PLAN_COMPLETES_AFTER_DEADLINE" in v
               for v in validate_decision(REQUEST, bad).violations)


def test_not_recommended_must_carry_no_payments():
    bad = {**ROW, "affordability_status": "not_affordable",
           "recommended_payment_method": "not_recommended"}
    assert any("PLAN" in v for v in validate_decision(REQUEST, bad).violations)


def test_plan_parser_round_trips_and_ignores_junk():
    assert parse_plan("none") == []
    assert len(parse_plan("2026-01-03:200|2026-02-03:300")) == 2
    assert parse_plan("garbage") == []


# --------------------------------------------------------------------------- #
# Evidence and currency rules
# --------------------------------------------------------------------------- #

def _row(**overrides):
    base = {"event_id": "e1", "user_id": "u1", "event_type": "expense",
            "description": "Invoice", "category": "other", "direction": "debit",
            "amount": "100", "currency": "EUR", "event_date": "2026-01-10",
            "settlement_date": "2026-01-10", "status": "settled",
            "linked_event_id": "", "flexibility": "fixed", "minimum_allowed_amount": ""}
    return {**base, **overrides}


def test_blank_amount_is_recovered_from_evidence_not_treated_as_zero():
    """The spec is explicit that a blank amount is not zero. With the amount read
    from its image, the event counts at that amount; with nothing recovered it
    is left out entirely rather than silently becoming a zero-value event."""
    rates = RateTable([])
    blank = _row(amount="")
    recovered = build_events([blank], "EUR", rates, amount_overrides={"e1": 250.0})
    assert len(recovered) == 1 and recovered[0].signed == -250.0
    unrecovered = build_events([blank], "EUR", rates, amount_overrides={})
    assert unrecovered == [], "a blank amount with no evidence must not become a zero event"


def test_foreign_currency_converts_at_the_settlement_date():
    """Cash moves on settlement, so that is the rate that applies - not the date
    the transaction was initiated."""
    rates = RateTable([
        {"rate_date": "2026-01-01", "from_currency": "USD", "to_currency": "EUR", "rate": "0.80"},
        {"rate_date": "2026-02-01", "from_currency": "USD", "to_currency": "EUR", "rate": "0.95"},
    ])
    row = _row(currency="USD", amount="100",
               event_date="2026-01-20", settlement_date="2026-02-03")
    [event] = build_events([row], "EUR", rates)
    assert round(-event.signed, 2) == 95.00, (
        f"converted at {-event.signed}: expected the settlement-date rate (0.95), "
        f"not the event-date rate (0.80)"
    )


# --------------------------------------------------------------------------- #
# Cache / determinism
# --------------------------------------------------------------------------- #

def test_content_key_is_order_independent():
    assert content_key("ns", "m", {"x": 1, "y": 2}) == content_key("ns", "m", {"y": 2, "x": 1})


def test_content_key_separates_namespaces_and_models():
    assert content_key("a", "m", {"x": 1}) != content_key("b", "m", {"x": 1})
    assert content_key("a", "m1", {"x": 1}) != content_key("a", "m2", {"x": 1})


def test_cache_round_trips_and_counts_hits():
    with CallCache(Path(tempfile.mkdtemp()) / "c.sqlite3") as cache:
        assert cache.get("k") is None
        cache.put("k", "ns", "m", {"v": 1}, {"input_tokens": 10})
        value, usage = cache.get("k")
        assert value == {"v": 1} and usage["input_tokens"] == 10
        assert cache.stats()["hits"] == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failure(s)'}")
    raise SystemExit(1 if failures else 0)
