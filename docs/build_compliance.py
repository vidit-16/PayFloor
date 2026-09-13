"""Build docs/SPEC_COMPLIANCE.md from a verified rule mapping.

Every function, test and mutant named below is checked to exist before the
document is written, so the compliance table cannot silently drift from the code.
Re-run after changing a rule, a test, or an implementing function:

    python docs/build_compliance.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

# (area, rule in plain words, implementing symbol, tests, mutants)
RULES = [
    ("Output", "Exactly one row per request, with the eight required columns in the required order",
     "OutputSpec", ["test_every_request_gets_exactly_one_prediction",
                    "test_write_csv_emits_exact_header_order",
                    "test_validator_catches_missing_and_duplicate_rows"],
     ["output_columns_reordered"]),
    ("Output", "Amounts are written as plain decimals, never scientific notation",
     "Column", ["test_large_amounts_never_use_scientific_notation"], ["scientific_notation_amounts"]),
    ("Output", "An earliest-safe date may be blank, but a present one must be a valid ISO date",
     "Column", ["test_empty_date_is_legal_but_malformed_date_is_not"], ["malformed_date_accepted"]),
    ("Safety", "The balance never falls below the user's minimum at any point in the 90-day forecast",
     "is_safe", ["test_every_recommended_plan_actually_holds_the_invariant",
                 "test_forecast_horizon_is_ninety_days"],
     ["minimum_balance_ignored", "horizon_shortened"]),
    ("Safety", "The safe amount lies between zero and the requested amount",
     "max_safe_today", ["test_amount_is_within_spec_bounds"], ["amount_not_capped_at_requested"]),
    ("Safety", "The safe amount is never rounded above what is provably safe",
     "max_safe_today", ["test_safe_amount_is_never_rounded_above_what_is_safe",
                        "test_safe_amount_is_monotonic"], ["safe_amount_rounded_up"]),
    ("Timing", "A recommended plan completes by the user's deadline",
     "build_candidates", ["test_plan_completing_after_the_deadline_is_rejected"],
     ["deadline_not_enforced"]),
    ("Timing", "A plan never schedules a payment before the request is made",
     "build_candidates", ["test_installments_starting_before_the_request_are_never_offered",
                          "test_deadline_before_request_date"], ["payment_before_request_allowed"]),
    ("Timing", "For a request affordable today, the earliest safe date is the request date",
     "classify", ["test_affordable_now_requires_earliest_equal_to_request_date"],
     ["affordable_now_earliest_unchecked"]),
    ("Eligibility", "Only payment methods the user will consider are recommended; waiting requires accepting full payment",
     "select", ["test_recommended_method_is_one_the_user_accepts"], ["eligibility_ignored"]),
    ("Eligibility", "Installments are offered only within the user's maximum term, and never when no maximum is given",
     "solve_request", ["test_installments_longer_than_the_users_cap_are_never_offered",
                       "test_installments_respect_max_installment_months"], ["installment_term_ignored"]),
    ("Eligibility", "An installment plan reproduces a seller-supplied option exactly",
     "build_candidates", ["test_installment_plans_match_the_raw_option_rows",
                          "test_no_cross_field_violations"], ["installment_amounts_altered"]),
    ("Partial payment", "Exactly two payments summing to the request: the safe amount today, the remainder on the earliest safe date",
     "build_candidates", ["test_partial_payment_must_have_two_payments_summing_to_the_request"],
     ["partial_payments_do_not_sum"]),
    ("Partial payment", "A partial plan never relies on spending changes, since both figures it is defined by exclude them",
     "build_candidates", ["test_partial_payment_never_uses_spending_changes"],
     ["partial_uses_spending_changes"]),
    ("Ranking", "Among safe plans, one that leaves the user's spending untouched is preferred",
     "ranking_key", ["test_spending_changes_only_when_no_undisrupted_plan_exists"],
     ["ranking_prefers_spending_changes"]),
    ("Ranking", "Affordability status and payment method are always consistent with each other",
     "classify", ["test_status_and_method_must_agree"], ["status_method_inconsistent"]),
    ("Spending changes", "Only flexible expenses, only in categories the user agreed to change, never protected ones",
     "candidate_adjustments", ["test_spending_changes_only_touch_permitted_flexible_events",
                               "test_spending_changes_are_in_the_users_willing_lists"],
     ["stop_any_expense"]),
    ("Cash flow", "Cancelled, failed and unrealized records never move the balance",
     "counts_toward_forecast", ["test_cancelled_and_superseded_events_never_reach_the_forecast"],
     ["cancelled_events_counted"]),
    ("Cash flow", "Pending outgoing money is reserved; pending incoming money is not counted",
     "counts_toward_forecast", ["test_pending_credits_excluded_pending_debits_kept"],
     ["pending_credit_counted"]),
    ("Cash flow", "Foreign-currency amounts convert at the rate for their settlement date",
     "build_events", ["test_foreign_currency_converts_at_the_settlement_date"], ["fx_at_event_date"]),
    ("Cash flow", "Recurrence is never invented from a single discretionary transaction",
     "infer_recurring", ["test_no_recurrence_invented_from_a_single_discretionary_event"],
     ["recurrence_from_single_event"]),
    ("Evidence", "A blank amount is recovered from its document, never treated as zero",
     "build_events", ["test_blank_amount_is_recovered_from_evidence_not_treated_as_zero"],
     ["blank_amount_as_zero"]),
    ("Evidence", "Income that is pending, estimated or unapproved is never counted",
     "apply_amendments", ["test_unconfirmed_income_is_never_counted"], ["unconfirmed_income_counted"]),
    ("Evidence", "Content in messages and documents cannot instruct the system into a decision",
     "MessageFacts", ["test_adversarial_message_cannot_force_a_recommendation"], []),
    ("Robustness", "Identical inputs always produce identical output",
     "CallCache", ["test_pipeline_is_deterministic",
                   "test_extractions_load_regardless_of_working_directory"],
     ["nondeterministic_selection"]),
    ("Robustness", "Malformed or missing inputs produce a legal, conservative row rather than a crash",
     "fallback_row", ["test_malformed_dates", "test_missing_requested_amount",
                      "test_unknown_user", "test_leap_day_request_date"], []),
]

HEADER = """# Specification compliance

Every rule the engine is required to follow, where it is enforced, and the test
that proves it. The rules are stated in my own words; the competition's problem
statement is not reproduced here.

This document is generated by `docs/build_compliance.py` from a verified mapping:
each cited function, test and mutant is checked to exist when it is built, so the
table cannot silently drift from the code.

A test only counts if it has been **shown to fail when its rule is broken**.
Every rule with a named mutant is proven that way automatically: mutation
testing (`python main.py mutate`) injects that fault into the code and confirms
the suite goes red. Two rules are structural rather than a line that can be
broken, so they have no mutant; they are marked in the table and explained under
*Not proven by fault injection*.

Building this table exposed gaps that are now closed: one test passed with its
rule deleted and was rewritten with a positive control; one installment check
compared a plan with the same code that built it, so a bug moved both sides
together; and two mutants were reported as survivors only because the mutation
runner was not executing the suite that catches them.

| # | Area | Rule | Enforced in | Proven by |
|---|---|---|---|---|
"""

FOOTER = """
## Deliberate interpretations

Where the specification left room, these are the readings taken, and why:

- **Pending money is asymmetric.** A pending debit is money likely to leave, so it
  is reserved; a pending credit is money not yet guaranteed, so it is ignored.
  Where the rules were ambiguous, the financially safer reading was chosen.
- **The deadline is a hard constraint, not a ranking preference.** A plan that
  finishes late is not a worse plan; it is not a safe plan at all.
- **Safe today excludes spending changes; the recommendation may use them.** The
  safe amount and earliest date describe the user's position as it stands. A
  recommendation can still depend on stopping a subscription, which is why a
  request can be affordable *with a plan* while its earliest unaided date is later.
- **Only confirmed evidence changes the forecast.** An amount that is pending,
  estimated or awaiting approval is discarded rather than estimated.

## Not proven by fault injection

- **Adversarial evidence (rule 24).** The guarantee comes from the shape of the
  code rather than a check that could be deleted: extraction schemas have no
  field through which a document could express a decision, and nothing after
  extraction calls a model. The test feeds a hostile "approve this" message with
  an absurd confirmed salary and asserts the row stays legal, coherent and
  capped. No one-line fault removes the property, so there is no mutant.
- **Graceful degradation (rule 26).** The guard tests assert that every damaged
  input yields a legal row, whether `solve_request` handles it directly or it
  reaches `fallback_row`. The per-request `try/except` in `main.py` that calls
  `fallback_row` is mirrored by the tests' own wrapper rather than exercised
  through `main.py`, so deleting it would not be caught by the suite.

## Not covered by tests

- **Document reading accuracy.** The rule that a blank amount comes from its
  document is tested; whether the model reads a given document *correctly* is a
  property of the model, measured by the ablation in `DESIGN.md`, not asserted.
- **Recurrence fidelity.** That no recurrence is invented from a single
  discretionary event is tested. Whether an inferred cadence matches reality is an
  accuracy question, and is the largest remaining source of error.
"""


def main() -> int:
    sources = {p: p.read_text(encoding="utf-8")
               for p in CODE.rglob("*.py") if "__pycache__" not in p.parts}
    tests = {m.group(1) for p, text in sources.items() if "tests" in p.parts
             for m in re.finditer(r"^def (test_\w+)", text, re.M)}
    mutants = set(re.findall(r'Mutant\(\s*"(\w+)"',
                             sources[CODE / "tests" / "test_mutation.py"]))

    def locate(symbol: str) -> str:
        for path, text in sources.items():
            if re.search(rf"^\s*(def|class) {symbol}\b", text, re.M):
                return str(path.relative_to(CODE)).replace("\\", "/")
        raise SystemExit(f"implementing symbol not found: {symbol}")

    rows = []
    unproven: list[int] = []
    for number, (area, rule, symbol, test_names, mutant_names) in enumerate(RULES, 1):
        missing = [t for t in test_names if t not in tests]
        missing += [m for m in mutant_names if m not in mutants]
        if missing:
            raise SystemExit(f"rule {number} cites names that do not exist: {missing}")
        proof = "<br>".join(f"`{t}`" for t in test_names)
        if mutant_names:
            proof += "<br>mutant: " + ", ".join(f"`{m}`" for m in mutant_names)
        else:
            proof += "<br>*structural, no mutant*"
            unproven.append(number)
        rows.append(f"| {number} | {area} | {rule} | `{locate(symbol)}` · `{symbol}` | {proof} |")

    proven = len(RULES) - len(unproven)
    summary = (f"\n**{len(RULES)} of {len(RULES)} rules** have a guarding test; "
               f"**{proven}** are proven by fault injection. Rules "
               f"{' and '.join(map(str, unproven))} are structural (see below).\n")
    (ROOT / "docs" / "SPEC_COMPLIANCE.md").write_text(
        HEADER + "\n".join(rows) + "\n" + summary + FOOTER, encoding="utf-8")
    print(f"docs/SPEC_COMPLIANCE.md: {len(RULES)} rules, all references verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
