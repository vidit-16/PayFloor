# Architecture

How the engine is put together. For *why* it is built this way — the decision
logic, the calibration, and the alternatives that were measured and rejected —
see [`DESIGN.md`](DESIGN.md).

## Shape

The engine is a **batch decision pipeline**, not a service and not an agent loop.
It reads a set of CSV files, makes one decision per request, and writes a CSV.

It has two halves that never mix:

| Half | Job | Uses a model? | Deterministic? |
|---|---|---|---|
| **Evidence extraction** | Turn unstructured text and images into structured facts | Yes | Cached, so reruns are |
| **Decision** | Reconstruct cash flow, forecast, search plans, rank, explain | **No** | Yes |

The boundary between them is a set of Pydantic schemas. Facts cross it; decisions
do not. There is no field in any extraction schema through which a document could
express a recommendation.

```
                  ┌────────────────── evidence extraction ──────────────────┐
  messages.csv ──►│ extract_message ─► MessageFacts                         │
  images.csv   ──►│ extract_image   ─► ImageFacts        (cached per input) │
  event labels ──►│ run_descriptions ─► continuity per description          │
                  └──────────────────────────────┬──────────────────────────┘
                                                 │ structured facts only
  financial_events.csv ─┐                        ▼
  exchange_rates.csv ───┼──►  build_events ─► infer_recurring ─► apply_amendments
  financial_profiles ───┘         │                                      │
                                  └──────────────► project ◄──────────────┘
                                                     │  Forecast (90 days)
                  ┌──────────────────────────────────┼───────────────────────┐
                  │  max_safe_today       earliest_full_payment_date         │
  request_        │            build_candidates ─► select ─► classify        │
  payment_ ───────┼──►           (safety + deadline + eligibility + ranking) │
  options.csv     │                          explain                         │
                  └──────────────────────────────────┬───────────────────────┘
                                                     ▼
                              validate_decision ─► OutputSpec ─► output.csv
```

## One request, end to end

`solve_request` in `pipelines/september2026.py` is the whole path for one row.

1. **Load the user.** Profile, home currency, current balance, minimum cushion,
   accepted payment methods, willingness to stop or reduce spending.
2. **Normalise events** — `build_events`. Drops records that never move cash
   (`counts_toward_forecast`: cancelled, failed, unrealized, non-cash, pending
   credits), recovers blank amounts from document evidence, converts every
   amount into the home currency at its settlement-date rate, and signs it.
3. **Infer recurrence** — `infer_recurring`. Groups spending by category and
   income by commitment, derives each flow's cadence, and corrects it with what
   the transaction descriptions say (a flow that ended is not projected).
4. **Apply evidence** — `apply_amendments`. Only confirmed, quantified facts from
   messages change the picture; anything pending or estimated is discarded.
5. **Forecast** — `project`. Projects every flow 90 days forward on calendar
   dates, adds known future-dated events, and returns a `Forecast`: a balance path
   that can test any candidate payment schedule against the minimum.
6. **Compute the two quantities** — `max_safe_today` (the forecast minimum less the
   cushion, capped at the request, floored to the cent) and
   `earliest_full_payment_date`. Both on the unadjusted forecast.
7. **Search** — `build_candidates` enumerates full payment today, waiting, a
   partial plan, and every seller-supplied installment option, each also under
   the spending changes the user permits. Unsafe or late plans are discarded
   here, not ranked.
8. **Choose** — `select` removes methods the user will not consider and takes the
   minimum of the six-part `Candidate.ranking_key`; `classify` maps the winner to
   an affordability status and payment method.
9. **Explain** — `explain` renders the sentence from the winning plan's own
   numbers and the named expenses it changes.
10. **Validate** — `validate_decision` checks every cross-column relationship the
    specification states; `OutputSpec` coerces and checks each column; the row is
    written.

## Modules

| Module | Responsibility |
|---|---|
| `code/main.py` | Command-line entry point: `run`, `score`, `validate`, `test`, `verify`, `mutate`, `extract`, `calibrate`, `sanity` |
| `pipelines/september2026.py` | `Dataset` loading and indexing; recurrence inference; amendments; projection; per-request wiring |
| `pipelines/sept_state.py` | `Event`, `Recurring`, `Forecast`, `Adjustment`; event normalisation; `RateTable` FX lookup |
| `pipelines/sept_solver.py` | `PaymentOption`, `Candidate`; safe-amount and earliest-date computation; candidate search; ranking; classification; explanations |
| `pipelines/sept_validate.py` | Cross-field decision validation, returning named violations rather than a boolean |
| `pipelines/sept_extract.py` | Extraction schemas and system prompts for messages and images |
| `orchestrate/schema.py` | `OutputSpec` / `Column`: the output contract, coercion and per-column validation |
| `orchestrate/llm.py` | OpenAI-compatible model client, structured output, retries, `UsageLedger` |
| `orchestrate/cache.py` | `CallCache`: content-addressed SQLite cache of every model call |
| `orchestrate/config.py` | Settings from environment variables and `.env` |
| `evaluation/calibrate.py` | Joint fit of the two projection-bias scales, with leave-one-out validation |
| `data/generate_synthetic.py` | Seeded synthetic dataset with planted edge cases and matching extractions |

## Core data types

| Type | Meaning |
|---|---|
| `Event` | One cash movement: date, signed home-currency amount, category, status, flexibility |
| `Recurring` | An inferred repeating flow: cadence, amount, last occurrence, flexibility |
| `Forecast` | A 90-day balance path; `is_safe(payments)` and `min_balance(payments)` test a schedule |
| `Adjustment` | A permitted spending change: stop a flow, or reduce it to a floor |
| `PaymentOption` | A seller-supplied way to pay: method, amount, count, first date, frequency, fee |
| `Candidate` | A concrete plan under consideration, with its `ranking_key` |

## Properties the structure guarantees

- **No model on the decision path.** Everything from `build_events` onward is pure
  Python. The same inputs always produce the same row.
- **One forecast, one source of truth.** The safe amount, the earliest date, the
  plan search and the invariant tests all interrogate the same `Forecast` object;
  nothing recomputes future balances independently.
- **Safety filters before ranking.** A plan that breaches the minimum or misses
  the deadline is removed in `build_candidates`. Ranking only ever orders plans
  that are already safe, so no ranking weight can rescue an unsafe one.
- **Extraction is keyed on content, not on requests.** Each message, image and
  distinct description is read once. A solver change costs no model calls.
- **Degrades rather than fails.** A malformed row yields a legal conservative
  output (`fallback_row`); missing extractions print a warning and fall back to
  the deterministic core rather than silently producing worse answers.

## Dependencies

| Package | Used for |
|---|---|
| `pydantic` | Extraction schemas and validated model output |
| `openai` | The model client — only reached by `main.py extract` |
| `tenacity` | Bounded retries on transient model-call failures |
| `python-dotenv` | Loading `.env` for the extraction step |

The decision pipeline itself needs only the standard library plus `pydantic`.
