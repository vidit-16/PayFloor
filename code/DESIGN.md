# Buy or Wait? — System Design

## The central claim

This task looks like classification and is not. The specification defines
affordability as an **invariant over a forecast**:

> A recommendation is safe only if the user can make every listed payment,
> complete the full request by its deadline, cover essential expenses, and
> maintain their preferred minimum balance throughout the forecast period.

An invariant plus an explicit 6-level tie-breaker means the correct answer is
**computable**, not predictable. So the model is never asked what to recommend.
It is asked only to read the unstructured evidence — multilingual payroll
messages, document images — and return numbers. Everything downstream is a
deterministic search.

```
profiles + events + FX + payment options        messages + images
              |                                        |
              |                                   model extracts
              |                                   structured facts
              v                                        v
        financial state reconstruction  <----  amendments / amounts
                          |
                          v
                90-day balance forecast
                          |
              +-----------+-----------+
              |                       |
      amount_safe_to_pay      earliest_date_for_full_payment
              |                       |
              +-----------+-----------+
                          v
                 candidate plan enumeration
             (full / partial / each installment option,
              with and without permitted spending changes)
                          |
                  safety filter (invariant)
                          |
                  eligibility filter (user's accepted methods)
                          |
                  6-level ranking -> winner
                          |
                          v
          output.csv  +  templated decision_explanation
```

## Why this split

Three consequences, each mapping to something scored:

* **Accuracy.** `amount_safe_to_pay` is a continuous number. No amount of
  prompting produces it; a forecast plus one subtraction does.
* **Consistency.** Two users in the same position get the same answer, because
  the answer is a function, not a generation.
* **Defensibility.** Every output traces to a rule that fired and a forecast
  that can be printed. "Why this plan?" has an exact answer.

## Decision logic

### 1. Which events are real

| Condition | Treatment | Why |
|---|---|---|
| `status` in cancelled / failed / unrealized | drop | Spec: ignore these outright |
| `direction = non_cash` | drop | Investment valuations never move the balance |
| `status = pending` and `direction = credit` | **drop** | Spec: ignore pending credits — money not yet guaranteed |
| `status = pending` and `direction = debit` | **keep** | Money likely to leave; the spec's "financially safer interpretation" |
| `status` in settled / scheduled | keep | Real, or confirmed future |

A blank `amount` is never zero: it is recovered from the linked image.

### 2. Which flows recur

There is no recurrence column, so it is inferred. Grouping is by **category**,
not description, because the data generates a category at a cadence while
varying the merchant name — `groceries` arrives every ~10 days under seven
different descriptions. A group qualifies when its median gap matches a known
cadence; monthly flows are projected on the **calendar month** (rent lands on
the 3rd, not every 30 days), sub-monthly ones on their median gap.

### 3. How evidence amends the picture

A message may raise a salary, confirm a temporary reduction, cancel a charge, or
report a bonus that is *not yet approved*. Only **confirmed, quantified** facts
are applied. Anything pending or conditional is discarded rather than estimated,
because counting unapproved income is precisely the error that makes an unsafe
plan look safe. Recurring amendments that supersede history replace the
projected amount for that category from their effective date.

### 4. The two quantities

With the payment on `request_date` shifting every later point down equally:

```
amount_safe_to_pay = clamp(baseline_90d_min − minimum_balance_to_keep,
                           0, requested_amount)
earliest_date_for_full_payment
    = first date d where paying the full amount on d keeps the
      forecast ≥ minimum_balance_to_keep through the horizon
```

Both are computed **before** optional spending changes, per the spec.

### 5. Choosing the plan

Candidates: full payment today; partial payment (exactly two payments, summing
to `requested_amount`); every installment option supplied for the request; wait;
and each of those again under permitted spending changes. Filter by the safety
invariant, then by `payment_methods_user_will_consider`, then rank:

1. completes by `desired_completion_date`
2. needs no spending changes
3. lowest total paid — *discriminating*: 515 of 790 options carry a financing fee
4. starts earlier
5. fewer payments
6. lowest `payment_option_id`

`not_recommended` is the fallback when nothing eligible is safe.

### 6. Spending changes

Only flows whose `flexibility` permits it **and** whose category appears in the
user's willing-to-reduce / willing-to-stop lists. Stop and reduce are mutually
exclusive on the same event; at most three changes.

## Safety posture

Message and image content is untrusted evidence. The extraction prompts state
this explicitly and the schemas have no field through which a document could
express a recommendation — the strongest thing a malicious message can do is
assert a number, which then has to survive the same forecast as everything else.
Embedded instructions cannot reach the decision because the decision is not made
by a model.

## What is deliberately not built

* No vector store or retrieval — the joins are keyed, so BM25/embeddings would
  add failure modes and no accuracy.
* No model-generated prose in `decision_explanation`; the golden explanations are
  templated, so templates reproduce the register exactly and remove variance
  from a scored column.
* No asset-price modelling for investment requests: the spec says affordability
  only.

## Reproducibility

Every model call is content-hash cached in SQLite, so a rerun after a solver
change costs nothing and returns byte-identical extractions. Determinism is a
property of the deterministic layer, not of a temperature setting; the solver is
pure, so the whole pipeline is reproducible end to end.
