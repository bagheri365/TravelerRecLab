# ADR-022 — Add a GRU while keeping v0.3 input and ranking contract

**Status:** Proposed for experiment; acceptance depends on validation/test evidence.

## Context

Our learned mean-pool model reached 0.466170 tie-neutral test MRR after three epochs, while a no-training last-item heuristic reached 0.504871. The next objective is to ask whether learned *event order* offers incremental value, without confounding that question by adding hotel content or search context at the same time.

## Decision

Implement a single-layer GRU over the **same ordered hotel-ID interactions** used by mean pooling. Reuse its data loading, train-only vocabulary, padding/unknown token policy, mini-batch construction, candidate embeddings, target-index softmax objective, split boundaries, and tie-neutral metrics. Select the best checkpoint only on validation. Keep position, hotel metadata, time gaps, and action features out of the initial GRU.

## Alternatives and trade-offs

- **Immediate GRU+metadata+action features:** potentially better utility but prevents attribution of any gains to event order. Deferred.
- **Transformer:** additional parallel attention complexity without an initial recurrent reference. Deferred.
- **Recency-weighted learned pooling:** a valuable lower-complexity neural control but outside this single-architecture increment; consider after reviewing GRU outcomes.
- **True GRU4Rec session-parallel objectives:** materially different training/evaluation protocol; would not be feature-and-loss matched with v0.3.

## Acceptance criteria

The implementation runs deterministically on a synthetic sample; padded events do not alter the state; reversing real hotel order changes the GRU output; empty hotel histories and unknown IDs are handled. Real-data numerical improvement is a hypothesis, not part of code acceptance.

## Cautions

A parameter-matched architecture comparison has **not** been established: the GRU contains recurrent weights in addition to embedding parameters, and both models may need longer training and matched tuning budgets. Evaluation candidates are reconstructed from clickout rows, and the dataset spans only six days. Do not overclaim cold-start capability or deployment relevance.
