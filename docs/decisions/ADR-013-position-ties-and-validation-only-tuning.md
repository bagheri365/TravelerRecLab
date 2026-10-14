# ADR-013 — Audit position bias and score ties before selecting a sequence model

**Status:** Accepted as evaluation control; model superiority not yet established.

## Observation

In the original v0.2 benchmark (223,074 test cases), last-item MRR was 0.5980, while recency-weighted MRR was 0.5658. Both used input-index tie-breaking. Repeated target behavior was common in EDA.

## Decision

1. Score the logged candidate-list order as a **position-only control**.
2. Report both existing logged-order tie-breaking and **tie-neutral expected MRR / Recall@5 / NDCG@5**. With tied scores spanning ranks `a+1 .. a+k`, average each metric over these `k` ranks, rather than crediting the logged candidate order.
3. Break down metrics by previously seen/unseen target and history length (1–5, 6–20, 21+).
4. Compare several recency decay values. Select the decay **only on validation all-segment tie-neutral MRR**; the test results are diagnostic, not a source of hyperparameter choice.

## Why / alternatives

Candidate order may reflect an existing ranking system, so optimistic metrics can result when a model assigns many identical scores. We keep logged-order metrics for comparability but add neutral metrics to estimate improvement not inherited from tie order. Random tie resolution would add sampling variance; analytic expectations are deterministic.

## Caveats

- Position is not evidence of causality; exposure logging occurred on click-out rows, not separately before decision time.
- The "unseen" segment means no matching **hotel-typed** reference in the truncated history, not never seen anywhere by the traveler.
- Current action weights are heuristics. Action-weight tuning and uncertainty intervals are future work.
- Distinct events in a session are not statistically independent; confidence intervals should cluster by session/user.
- Compare neural models on the same candidates, population, splits, and tie protocol.

## Interview response

> My first baseline appeared strong, but many score ties inherited the original candidate order. I therefore measured a position-only control and tie-neutral expected metrics, segmented results by previous hotel engagement, and tuned recency only on the validation period. That helped distinguish real behavioral value from the logged ranking's influence before introducing a GRU.
