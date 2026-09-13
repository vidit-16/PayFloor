# Token Usage and Cost Report

Final full-dataset run producing `dataset/output.csv` (250 requests).

## Provider and models

| Provider | Model | Role |
|---|---|---|
| OpenAI | `gpt-5-mini` | Message fact extraction, document-image reading, event-description classification |

A single model is used for all three extraction roles. No model is invoked in the
decision path: the solver is deterministic, so the model count below is driven by
the size of the evidence corpus, not by the number of requests.

## Per-model totals

| Stage | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|
| Message facts (215 messages) | 215 | — | — | — |
| Document images (16 images) | 16 | — | — | — |
| Message + image subtotal | 231 | 173,530 | 174,476 | 0.3923 |
| Event-description classification (164 distinct) | 164 | 48,391 | 46,171 | 0.1044 |
| **Total** | **395** | **221,921** | **220,647** | **0.4967** |

## Per-request figures

| Metric | Value |
|---|---|
| Requests processed | 250 |
| Model calls per request | 1.58 |
| Input tokens per request | 888 |
| Output tokens per request | 883 |
| Total tokens per request | 1,771 |
| Cost per request | USD 0.0020 |

## Pricing assumptions

`gpt-5-mini` at USD 0.25 / 1M input tokens and USD 2.00 / 1M output tokens
(list price at time of writing). Output tokens exceed the visible response
length because `gpt-5-mini` is a reasoning model and reasoning tokens bill as
output.

## Model comparison

A full extraction run was also performed with `gpt-5` to justify the model
choice empirically:

| model | calls | input | output | cost | mean accuracy |
|---|---|---|---|---|---|
| `gpt-5-mini` | 395 | 221,921 | 220,647 | **$0.4967** | 71.4% |
| `gpt-5` | 395 | 222,180 | 219,546 | $3.0722 | 71.4% |

Identical accuracy on every scored column at one sixth the cost, so
`gpt-5-mini` ships. Full reasoning in `DESIGN.md`.

## Efficiency notes

Extraction is keyed on the *content*, not the request. The 215 messages, 16
images and 164 distinct event descriptions are each read exactly once and cached
by a SHA-256 of (model, system prompt, payload) in SQLite, so:

- a request referencing already-seen evidence costs zero additional calls;
- re-running the full dataset after any solver change costs **USD 0.00**;
- the 25,342 financial-event rows are classified through only 164 calls, because
  descriptions repeat across users.

A naive design that sent each request's full context to a model would cost
roughly 250 calls with far larger prompts. Concurrency is bounded at 6 in-flight
requests as the primary TPM/RPM control, with exponential-backoff retries.
