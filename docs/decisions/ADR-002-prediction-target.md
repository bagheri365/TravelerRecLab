# ADR-002 — Predict the next hotel click-out

**Status:** Provisional
**Date:** 2026-10-13

## Problem
What behavior will teach a session embedding useful hotel preference, without being dominated by high-volume but weaker actions?

## Alternatives
- Next arbitrary action: dense supervision but images dominate.
- Next distinct interacted item: more variety, but ambiguous action strength.
- **Next click-out hotel**: hotel-selection signal; lower frequency than image actions.
- Confirmed booking: strongest commercial endpoint, **not present** in Trivago.

## Decision
Use the clicked-out `reference` item as the primary label; use preceding actions as features. Analyze repeated click-outs and multiple targets within a session. Candidate lists on the target row are used as *offline evaluation candidate lists*, not as evidence of a separate prior impression.

## Evidence
1,586,586 click-outs of 15,932,992 events (~9.96%), while image interactions are 11,860,750 (~74.44%). These are dataset event totals, **not** the number of valid training examples.

## Trade-offs
Click-outs depend on position, exposure, available listings, and interface behavior; clicks may not imply booking. Excluding image actions from targets changes what embeddings learn. The choice could bias the model toward clickability.

## Verify before training
- Count valid click-out-target examples and candidate membership.
- Inspect histories at click-out events (including zero-history cases).
- Inspect repeated same-item targets and per-session weighting.
- Determine prediction-time interpretation of candidate lists and prices.
- Compare an alternative target in a controlled ablation.

## Interview explanation (60 seconds)
> Next click-out is a practical proxy for hotel interest in this dataset. I do not call it booking propensity. I still encode search, filters, and image actions as the context that leads to a click-out. The choice is provisional: I would validate whether it improves the downstream ranking objective.
