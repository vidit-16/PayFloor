# Scope

What the engine does, what it assumes, and — just as deliberately — what it does
not do.

## In scope

Given a user's financial history and a request to spend money, decide:

| Output | Meaning |
|---|---|
| `amount_safe_to_pay` | The most the user can pay today without ever breaching their minimum over 90 days |
| `affordability_status` | Affordable now · affordable with a plan · affordable later · not affordable |
| `recommended_payment_method` | Full payment · partial payment · installments · wait · not recommended |
| `payment_plan` | The dated payments that make up the recommendation |
| `earliest_date_for_full_payment` | The first date a single full payment becomes safe, or blank if never within 90 days |
| `spending_changes_needed` | Flexible expenses to stop or reduce, if the plan depends on them |
| `decision_explanation` | A sentence stating the recommendation and the figures behind it |

**Currencies:** EUR, USD, INR, IDR, ZAR, with dated conversion rates.

**Request types:** purchase, travel, education, family transfer, debt repayment,
investment, housing, emergency expense, other.

**Evidence handled:** transaction history; scheduled, pending, failed, cancelled
and unrealized records; seller payment options; free-text messages (salary
changes, cancellations, fees, pending bonuses); document images carrying amounts.

## Assumptions

- **A 90-day horizon.** Safety is judged over exactly the next 90 days. A plan that
  is safe for 90 days is treated as safe, even if a breach would follow on day 91.
- **The minimum cushion is a hard floor.** Touching it is allowed; going below it,
  by any amount at any time, is not.
- **Commitments recur on calendar dates.** Rent due on the 3rd is due on the 3rd of
  every month, not every 30 days.
- **History is representative.** Cadences and amounts are inferred from the
  transactions supplied; a change in circumstances not recorded anywhere in the
  data cannot be anticipated.
- **Only confirmed evidence counts.** Amounts that are pending, estimated or
  awaiting approval are discarded, not estimated. This is conservative by design.
- **Pending money is asymmetric.** Pending outgoing payments are reserved; pending
  incoming money is not counted until it settles.
- **Fixed exchange rates.** Conversions use the dated rate table provided; there is
  no live market data.

## Out of scope

- **Financial advice.** The engine determines whether a request is affordable under
  stated rules. It does not judge whether a purchase is wise, recommend
  investments, or predict asset prices.
- **Live data.** No banking connections, market feeds or live exchange rates.
- **Conversation.** It is not a chat assistant; there is no dialogue, and requests
  are evaluated independently.
- **Budgeting beyond the request.** It proposes spending changes only to make a
  specific request affordable, and only within the categories the user has agreed
  to change. It does not optimise a user's budget in general.
- **Model-driven decisions.** By design, no model decides anything. See
  [`DESIGN.md`](DESIGN.md) for why.

## Known limitations

- **The safe amount is approximate.** On the competition's labelled requests it
  was exact on 24% and within a median error of 6.6%. The largest remaining error
  comes from users whose income arrives as several interleaved streams, where the
  inferred cadence is imprecise.
- **Calibration rests on 25 labelled examples.** Two projection-bias scales were
  fitted jointly and pass leave-one-out validation, but that is a small sample; a
  different population could shift the right values. The uncalibrated model is
  within 1.7 points.
- **Document reading depends on the model.** The rule that a blank amount comes from
  its document is enforced and tested; whether a particular document is *read*
  correctly is a property of the model, not of the engine.
- **No ground truth for synthetic data.** The bundled dataset exercises every code
  path, but because it is generated it cannot measure accuracy. Reported accuracy
  comes from the competition's labelled set, which is not redistributed.

## Data boundaries

The engine was built for HackerRank Orchestrate (September 2026). The competition's
dataset, problem statement and derived model outputs are not part of this
repository. Everything needed to run it is generated locally by
`data/generate_synthetic.py`.
