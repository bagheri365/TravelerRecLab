# ADR-003 — Causal, session-local history prefixes

**Status:** Accepted
**Date:** 2026-10-13

## Problem
Prevent a representation from encoding the future action used as its label.

## Options
- Aggregate full traveler history: easy, but leaks future behavior into past training points.
- **Prefix through strictly earlier session events**: enforces a local causal boundary.
- Multi-session prefix: potentially better long-term personalization; requires separate identity and as-of design.

## Decision
v0.1 groups events by `(user_id, session_id)` and orders by `(timestamp, step, source row order)` before generating pre-click-out history prefixes. The target row is excluded from `history`. Keep missing-context fields missing rather than filling from future rows.

## Caveats and verification
Source row order breaks ties deterministically; it does not establish physical causality for tied timestamps. Candidate lists/prices attached to the target click-out event are **not** assumed to be available at an earlier history timestamp. The current v0.1 script is an offline modeling data builder, not a point-in-time feature store.

Test target exclusion, session isolation, duplicate/tied event ordering, candidate membership, and future-event contamination. Cross-session preference learning is a later isolated experiment.

## Interview explanation (60 seconds)
> Every training example has a decision point, and its representation is computed only from earlier ordered actions. I keep a clear boundary between historical inputs and labels. Logging-time availability can differ from event time, so online leakage still needs an as-of audit beyond simply sorting timestamps.
