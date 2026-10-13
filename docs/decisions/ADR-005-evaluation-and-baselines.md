# ADR-005 — Chronological evaluation, with strong non-sequential baselines

**Status:** Accepted (protocol direction); metric implementation pending
**Date:** 2026-10-13

## Problem
Show that learned sequence structure adds value rather than just model capacity or recency.

## Options
- Random row split: easy but risks future leakage and correlated examples.
- **Chronological split by target time:** closer to deployment, but requires explicit boundary checks.
- Session/user holdout: useful complementary cold-start experiment, not equivalent to deployment over time.

## Decision
Use chronological train/validation/test windows, derive prefixes without future events, and compare popularity, last-item, mean pooling, action weighting, and recency weighting before GRU and Transformer. Use the same eligible candidate protocol and common context for fair comparisons.

**Primary planned metric:** MRR on held-out click-out examples. **Supporting:** Recall@K, NDCG@K, slice metrics, p95 latency. Define whether ranking uses target-row logged candidates or full inventory; don't compare the two as if interchangeable. Split by target time, ensure no future interactions appear in histories, and avoid training item representations on validation/test future events.

## Unresolved
- Target click-out candidate list is logged on the label row; it cannot be claimed to be known pre-click. Offline candidate ranking may still be valid, but it evaluates a conditional task.
- Need a consistent treatment for items not among logged candidates.
- Same traveler can appear across splits; avoid future-to-past feature or embedding leakage.
- Assess sample weighting so sessions with many click-outs do not dominate.

## Interview explanation (60 seconds)
> A sequential model must beat a tuned recency/action aggregation using identical temporal splits, candidate sets, and features. I'd prioritize retrieval MRR and NDCG, but only deploy if that translates into useful online outcomes for acceptable latency and cost.
