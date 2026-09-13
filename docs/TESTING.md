# Testing

How the engine is verified, and how the verification itself is verified.

```bash
cd code
python main.py verify     # the full gate, in the order a failure is cheapest to find
python main.py mutate     # slow: re-runs every suite once per injected fault
```

CI runs the same sequence on every push (`.github/workflows/ci.yml`); mutation
testing runs on demand because it re-executes the suite 26 times.

## The suites

| Suite | Tests | Question it answers |
|---|---:|---|
| `tests/test_harness.py` | 19 | Is the output contract honoured, and are the cross-column rules enforced? |
| `tests/test_resilience.py` | 17 | Does damaged, missing or hostile input produce a legal, conservative row rather than a crash? |
| `tests/test_invariants.py` | 18 | Do the specification's properties hold on every one of the 250 requests? |
| `tests/test_mutation.py` | 26 mutants | Do the three suites above actually fail when a rule is broken? |

**54 tests, 26 mutants.** Every specification rule is mapped to its guarding test
and mutant in [`SPEC_COMPLIANCE.md`](SPEC_COMPLIANCE.md).

### Unit and contract — `test_harness.py`

Fast, focused checks on single functions: exact header order, plain-decimal
amounts, ISO dates, status/method consistency, partial plans summing to the
request, deadline enforcement, currency conversion at the settlement date,
recovery of a blank amount from its document, and the content-hash cache that
makes model calls reproducible.

### Guards — `test_resilience.py`

Inputs the real data never contained: blank, negative and non-numeric amounts;
malformed dates; a deadline before the request; 29 February; unknown users; users
with no history, no balance or no acceptable payment methods; requests with no
seller options; and extractions loading identically from any working
directory. The requirement is not a good answer — a damaged row cannot have
one — but a legal, conservative one.

It also holds the two tests that build a scenario from scratch because the data
cannot provide one: installment plans longer than the user's cap, and plans that
start before the request. The synthetic data never has such a plan win, so a
missing check changes no output row; these tests construct the case directly.

### Invariants — `test_invariants.py`

Properties asserted over every request, independent of which answer is correct:

- every recommended plan keeps the balance at or above the profile's minimum,
  re-simulated against the floor itself rather than through the solver's own
  safety function;
- the safe amount lies within bounds, is never rounded up, and every smaller
  amount is safe too;
- cancelled and superseded records never reach the forecast; pending debits are
  kept and pending credits dropped; no recurrence is invented from one purchase;
- installment plans reproduce a seller option, checked against the raw CSV rather
  than the code that built the plan;
- spending changes touch only flexible expenses in categories the user agreed
  to, and appear only when no undisrupted plan exists;
- identical inputs produce identical output.

## Testing the tests

A passing suite proves only that nothing it checks is broken. Mutation testing
asks the harder question: *if this rule were broken, would anything notice?*

`test_mutation.py` copies the code, injects one fault, runs all three suites, and
restores it — 26 times. Each fault is a plausible real bug: counting pending
income, dropping the minimum-balance floor, shortening the horizon, rounding the
safe amount up, recommending a payment method the user rejects, converting
currency on the wrong date, picking among eligible plans at random.

**Result: 25 of 25 non-equivalent mutants killed.**

The 26th, `spending_changes_unrestricted`, is an *equivalent mutant*: it lets the
solver propose stopping expenses the user never agreed to, but ranking rule 2
prefers plans with no spending changes, so those candidates never win. Diffing
all 250 output rows under the mutation shows zero changes. No test can kill a
fault with no observable effect, so surviving is the correct result — and the
stronger version of the same fault, `stop_any_expense`, is killed.

### What mutation testing found

Each row is a gap between what the suite appeared to prove and what it did:

| Finding | Cause | Fix |
|---|---|---|
| First run killed 6 of 10 | Nothing checked the horizon length, rounding direction, accepted methods or willingness lists | Five targeted tests |
| Invariant passed with the floor removed | It checked plans with `Forecast.is_safe()`, the function the mutant broke | Compare against the profile's minimum directly |
| Unconfirmed-income test passed with the check deleted | It used a request already capped at the requested amount, where more income cannot change anything | Use a binding request, with a positive control proving a confirmed claim *does* change it |
| Altered installment amounts changed 11 rows, no test failed | The validator compared a plan with `option.schedule()`, which built it | Rebuild schedules from the raw CSV |
| Two mutants reported as survivors | The runner executed two of the three suites | Run all three |
| Longer-than-cap installments undetected | Never occurs in the data | Scenario test with a positive control |

The lesson running through the table: **a mutation score is a property of the
tests and the data together.** A fault can go uncaught because a test is
circular, because it is vacuous on its input, or because the data never
exercises the path. Each needs a different fix, and only the positive control —
first proving the test *can* see the effect — distinguishes a real guard from a
coincidence.

## Beyond the suites

| Check | Command | What it catches |
|---|---|---|
| Output validation | `python main.py validate` | Schema and cross-field violations in an existing `output.csv` |
| Sanity report | `python main.py sanity` | Degenerate columns, amounts above the request, dates outside the window, explanations under six words; distribution parity against labelled samples when present |
| Calibration | `python main.py calibrate` | Leave-one-out validation of the two projection scales |

## Synthetic data as a test instrument

`data/generate_synthetic.py` is seeded and plants the cases real data rarely
contains: cancelled authorisations, pending debits and credits, failed payments,
employment ending mid-history, leave gaps, foreign-currency subscriptions, and
messages that try to instruct the system. Running on it exposed two solver
defects the competition data never triggered — see
[`DESIGN.md`](DESIGN.md#what-synthetic-data-found-that-the-real-data-didnt).

Because it is generated, it has no independent ground truth. It measures
correctness of *rules*, never accuracy of *answers*; `main.py score` says so
rather than printing a circular number.
