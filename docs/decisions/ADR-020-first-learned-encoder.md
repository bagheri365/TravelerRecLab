# ADR-020 — Start representation learning with a trainable mean-pooled hotel encoder

**Status:** Provisional; run pending.

## Context
The ID-history baselines are highly effective for immediate continuation but weak on new-to-history hotels. The position + last-item hybrid obtained test MRR 0.5980, but used logged candidate order and regressed in earlier-hotel return and new-after-exploration segments. We are explicitly moving into trainable representation models while retaining this benchmark caveat.

## Decision
Start with tied hotel-ID embeddings, mean pooling of preceding hotel actions, and candidate-set cross-entropy. Use a train-only item vocabulary and UNK fallback. Report tie-neutral metrics and unknown-target rate. Keep candidate position out of this model. This is the simplest learned hotel-history representation and a feature-controlled precursor to GRU/causal Transformer comparisons.

## Alternatives rejected for the first experiment
- **GRU immediately:** possible, but confounds learned embeddings and event-order learning; use mean pooling as a neural control first.
- **Item metadata encoder:** better cold-item generalization, but requires a separately verified metadata join and extra features; add later symmetrically to mean pooling and GRU.
- **Original candidate position:** strong ranking baseline, but not a pure behavioral representation and pre-click availability is unverified.
- **Full catalog retrieval:** requires a separate negative/candidate generation protocol, unlike the existing benchmark.

## What would change this decision?
Evidence of numerical instability, very high unknown-target prevalence, or failure to distinguish candidate IDs. After the first run, verify behavioral-pattern performance before inferring model value. An eventual GRU must use exactly matched item inputs and losses so any improvement is attributable to sequence modeling.
