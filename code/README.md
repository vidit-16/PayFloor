# Buy or Wait? — affordability decision agent

Decides, for each request in `dataset/requests.csv`, whether the user should pay
in full, pay partially, use a supplied installment option, wait, or not proceed.

## Quick start

```bash
cd code
pip install -r requirements.txt
python main.py run            # writes ../dataset/output.csv
```

**`run` needs no API key.** `data/generate_synthetic.py` writes the dataset and
matching model extractions into `extracted/`, so the decision pipeline is fully reproducible
offline. `extract` is the only command that calls a model.

## Commands

| Command | What it does |
|---|---|
| `python main.py run` | Produce `output.csv` for all 250 requests, validate it, print the contract report |
| `python main.py score` | Grade against labelled samples, when the dataset has them (synthetic data does not) |
| `python main.py validate` | Re-check an existing `output.csv` (schema + cross-field) |
| `python main.py verify` | **The full test gate** — same sequence as CI |
| `python main.py test` | Unit, guard and invariant suites (54 tests) |
| `python main.py sanity` | Sanity and parity report on `output.csv` |
| `python main.py mutate` | Mutation testing: are the tests load-bearing? |
| `python main.py calibrate` | Refit the projection scales, with leave-one-out cross-validation |
| `python main.py extract` | Rebuild model extractions — **requires `OPENAI_API_KEY`** |

Add `--dataset PATH` / `--out PATH` to any command.

## Configuration

Only needed for `extract`. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`;
every other setting has a default and is documented in the example file.

Secrets are read from the environment only. No key is ever written to a file
that ships.

## Architecture

Full rationale in [DESIGN.md](../docs/DESIGN.md); module map in
[ARCHITECTURE.md](../docs/ARCHITECTURE.md). In one line: **the model describes,
deterministic code decides.**

```
messages / images / event descriptions        profiles, events, FX, options
             |                                            |
      model extraction                                    |
      (structured facts only)                             |
             +---------------------+----------------------+
                                   v
                    financial state reconstruction
                                   v
                        90-day balance forecast
                                   v
              candidate plans  ->  safety filter  ->  eligibility
                                   v
                     6-level deterministic ranking
                                   v
                    output.csv + templated explanation
```

The specification defines affordability as an invariant over a forecast plus an
explicit tie-breaker, which makes the answer computable rather than predictable.
No model is consulted in the decision path — so the same inputs always produce
the same recommendation, and evidence embedded in a message cannot influence a
decision it is not allowed to make.

## Layout

```
main.py                     entry point
pipelines/
  september2026.py          recurrence, amendments, projection -> row
  sept_state.py             events, FX, normalisation, the Forecast type
  sept_extract.py           model extraction schemas and prompts
  sept_solver.py            candidate generation, ranking, explanations
  sept_validate.py          cross-field decision validation
orchestrate/
  schema.py                 output contract + per-column validator
  cache.py                  content-hash SQLite cache (reproducibility)
  llm.py                    OpenAI-compatible backend, structured output, usage ledger
  config.py                 env-driven settings
evaluation/
  calibrate.py              joint fit of projection scales, with LOO
  usage_report.md           token and cost report for the final run
extracted/                  cached model output (messages, images, descriptions)
tests/
  test_harness.py           output contract, cross-field rules, cache
  test_resilience.py        malformed and adversarial input handling
  test_invariants.py        properties over all 250 requests
  test_mutation.py          proves the other suites catch real bugs
tools/
  sanity_report.py          sanity and parity check on output.csv
  package.py                builds code.zip, refuses to seal on a secret
  run_extract.py            message and image extraction
  run_descriptions.py       event-description classification
  profile_dataset.py        dataset profiler
```

## Testing

```bash
python main.py verify     # everything, in the order a failure is cheapest to find
```

Four suites — 54 tests plus 26 mutants. Full detail in
[TESTING.md](../docs/TESTING.md); every rule mapped to its test and mutant in
[SPEC_COMPLIANCE.md](../docs/SPEC_COMPLIANCE.md).


| suite | what it proves |
|---|---|
| `test_harness.py` | Output contract and cross-field rules. No dataset needed |
| `test_resilience.py` | Malformed, missing and adversarial input degrades to a legal conservative row rather than crashing |
| `test_invariants.py` | Properties over all 250 real requests |
| `test_mutation.py` | That the other three actually catch bugs |

`test_harness.py` covers the output contract and cross-field rules with no
dataset needed. `test_invariants.py` asserts properties over all 250 real
requests — bounds, determinism, no double counting, the pending-credit /
pending-debit asymmetry, no invented recurrence, spending changes never touching
protected categories, and the decisive one: **every recommended plan is
re-simulated independently of the solver that chose it**, so a disagreement
between planner and simulator fails loudly instead of shipping.

That test found a real defect the labelled samples could not — `amount_safe_to_pay`
was rounded to cents, and rounding up placed the plan a fraction below the
minimum balance it is defined by.

**Mutation score: 25/25 non-equivalent mutants killed.** Twenty-six deliberate
faults are injected one at a time — counting pending credits, ignoring the
minimum balance, shortening the horizon, dropping the deadline, recommending
rejected payment methods, converting currency on the wrong date — and the suite
must fail on each. The first run, with ten mutants, killed only 6; each survivor
was a genuine gap and gained a test. One mutant is *equivalent* — verified to
change 0 of 250 output rows, because ranking rule 2 means the extra candidates it
generates never win — so surviving is the correct result.

## Accuracy

Against the competition's 25 labelled samples (not redistributed here):

| column | exact |
|---|---|
| `recommended_payment_method` | 88% |
| `spending_changes_needed` | 88% |
| `payment_plan` | 84% |
| `affordability_status` | 80% |
| `earliest_date_for_full_payment` | 76% |
| `decision_explanation` | 60% |
| `amount_safe_to_pay` | 24% exact, 6.6% median error |
| **mean** | **71.4%** |

`decision_explanation` is scored here by exact string match against templated
gold, so the figure understates it — the rubric grades usefulness and
consistency.

## Cost

395 model calls, $0.50 total, $0.0020 per request — see
[evaluation/usage_report.md](evaluation/usage_report.md). Extraction is keyed on
content rather than per request, so the 250 requests, 25,342 events and 215
messages are covered by 395 calls, and re-running after a solver change costs
nothing.

## Known limitations

- The projection scales in `september2026.py` are fitted on 25 samples. Leave-one-out
  error matches in-sample, and the unscaled model still reaches 69.7%, but a
  different mix of users could shift the right values.
- Recurrence is inferred, not given. A user whose history is short or whose
  circumstances changed in a way no description records will forecast imprecisely.
- `amount_safe_to_pay` is exact on only 24% of samples, though median error is
  6.6% — the forecast is directionally right and quantitatively approximate.
  A residual analysis against the implied ground truth puts the median residual
  at 32 currency units: the forecast is near-exact on most rows and wrong on a
  handful of users whose income category interleaves several distinct streams.
  Three separate attempts to correct those rows are recorded in
  [DESIGN.md](../docs/DESIGN.md); each
  measured worse and was reverted.
