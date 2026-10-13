# ADR-006 — Start with 128-dimensional traveler vectors

**Status:** Provisional
**Date:** 2026-10-13

## Decision
Use dimension 128 as an initial experiment default, not an empirically optimized value.

## Alternatives
64 for efficiency; 256 for capacity; task-specific projections or factorized item embeddings.

## Trade-offs
More dimensions raise storage, candidate scoring cost, and risk of overfitting; fewer dimensions can bottleneck diverse intents. Ensure apples-to-apples model comparisons and report parameter counts.

## Test
Run dimension 64/128/256 with controlled budgets and report retrieval metrics, tail latency, state size, and training cost.

## Interview explanation (60 seconds)
> Embedding width is a quality-cost hyperparameter. I'd start with 128 to get working baselines and tune it against measured latency, memory, and recommendation lift rather than claiming it's intrinsically optimal.
