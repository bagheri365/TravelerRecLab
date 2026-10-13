# ADR-004 — Initial maximum history of 50 events

**Status:** Provisional
**Date:** 2026-10-13

## Problem
Trade off inference cost, memory, and truncation of useful history.

## Alternatives
20, 50, 100, or dynamic-length histories; later, separate long-term profile from short-term session.

## Decision
Use `--max-history 50` for the initial pipeline and benchmark 20/50/100 in controlled experiments.

## Evidence and limitation
All-session event-length median = 4, p90 = 41, p99 = 212, maximum = 3,522. This suggests 50 covers whole histories for at least 90% of *all* sessions, but it does **not** prove comparable coverage for training prefixes ending at click-outs. Measure the target-specific distribution.

## Revisit if
Long-history segments lose NDCG/MRR, runtime exceeds limits, or 100 events produces meaningful lift. Compare tail truncation rate and cost per request.

## Interview explanation (60 seconds)
> Fifty isn't a magic hyperparameter. It is a starting point motivated by the observed session-length distribution. I'd sweep sequence lengths and report both ranking quality and update latency, especially on users with long sessions.
