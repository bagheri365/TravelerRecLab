# ADR-018 — Test one position + behavior baseline before representation models

**Status: Provisional; awaiting validation/test evidence.**

## Context

The A.2/A.3 audits showed that next-click-out examples combine immediate continuations, returns, previously unreferenced target hotels, and no-hotel-reference histories. Earlier baseline results show last-item and upstream candidate position are strong in different slices. A stronger benchmark is needed before claiming benefit from traveler embeddings.

## Decision

Introduce an interpretable, validation-selected additive position-plus-last-hotel hybrid with fixed position normalization and explicit tie-neutral metrics. Keep all existing splits and eligibility rules. Evaluate each of the four A.2 behavioral patterns. Do not tune boost on test or add metadata features in this checkpoint.

## Why, alternatives, and risks

- Simpler than an end-to-end trained reranker; isolates the interaction between two already measured signals.
- Position-only (`boost=0`) is an internal exact control; v0.2.1 last-item remains the behavioral-only reference.
- A learned logistic ranker, content-based scorer, recency hybrid, and GRU are deferred because they add parameters/features and complicate attribution.
- The candidate position belongs to an existing ranking pipeline. Its presence in click-out-row logs does **not** prove it is available at the desired online scoring point.
- Ties at exact boost thresholds are handled by the v0.2.1 neutral metric, not implicitly credited to the original order.

## Revisit when

Full-data metrics and four-pattern slices are available. Decide whether this hybrid improves a candidate reranking benchmark, and whether it offers any benefit beyond exploitation of recorded position and immediate item identity. Do not conflate ranking lift with learned traveler preference representation.
