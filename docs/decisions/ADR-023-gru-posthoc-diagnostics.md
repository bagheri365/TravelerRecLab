# ADR-023 — Evaluate GRU behavior and order sensitivity before increasing model complexity

**Status:** Accepted for diagnostic implementation; empirical results pending.

## Context

The three-epoch hotel-ID GRU reached test MRR 0.515058 versus 0.466170 for learned mean pooling, on the same 223,074-example test set. However, the encoder has more parameters, and neither validation learning curve plateaued. Aggregate MRR alone obscures behavioral differences.

## Decision

Freeze the trained checkpoint. Add evaluation-only diagnostics for original versus seeded permuted hotel histories, and report the four existing retained-history behavior slices. No new features, retraining, hyperparameter selection, or test-optimized changes. A future architecture/tuning study requires a separately registered validation protocol and should not reinterpret test diagnostics as model-selection evidence.

## Why

This isolates whether the **existing** GRU predictions rely on interaction ordering without conflating the test with retraining. Group breakdowns identify where the current encoder performs well or poorly. The permutation preserves candidate list, labels and hotel counts, while tracking unchanged sequences.

## Limitations

Evaluation-time shuffling introduces a distribution shift and does not establish that order modeling caused the gain over mean pooling. Correlated session observations and the candidate-list provenance limit statistical and production claims.
