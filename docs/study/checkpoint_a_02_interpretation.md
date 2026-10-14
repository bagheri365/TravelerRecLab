# Checkpoint A.2 — Findings and interpretation (freeze before new models)

**Status:** observed distributions; data semantics still under audit. **Unit:** emitted eligible next-clickout example, not unique traveler or raw clickout. **Source:** `data/processed/behavior_patterns_a2.json` and `behavior_patterns_a2.md`. Calculations are descriptive, not causal. Percentages rounded.

## Measured behavior (1,276,216 examples)

| Mutually exclusive retained-history pattern | Count | Share |
|---|---:|---:|
| Immediate continuation (target = latest hotel reference) | 532,270 | 41.71% |
| Return to earlier hotel reference | 63,297 | 4.96% |
| Target unseen after other hotel references | 497,407 | 38.98% |
| No recognized hotel reference | 183,242 | 14.36% |

This taxonomy treats `search for item` as a hotel-referencing action **provisionally**. The term “immediate” refers to the last **hotel reference**, not necessarily the last event. Each example may contribute to several nonexclusive diagnostic dimensions; the four patterns themselves partition the examples.

The independent **retained** prior-clickout categories are: first 533,326 (41.79%), repeat 185,410 (14.53%), switch 557,480 (43.68%). Among examples containing any earlier clickout in retained history, switch accounts for 557,480 / (557,480 + 185,410) = **75.04%**; it does **not** imply booking rejection. For the new-hotel-after-exploration group, 399,905 of 497,407 (80.40%) are switches and 97,502 are first clickouts.

## Behavioral differences supported by cross-tabs

| Observation | Evidence | Interpretation, not a causal claim | Alternative / limitation |
|---|---|---|---|
| Frequent immediate continuation | 235,848 / 532,270 (44.31%) follow image interaction | Repeated hotel inspection is common near clickout | Logging volume / UI mechanics may dominate |
| New-hotel cases often follow clickouts | 294,891 / 497,407 (59.28%) have `clickout item` as last action | Multi-clickout sessions deserve separate study | Clickout is not a purchase or a rejection |
| No-hotel-history cases contain search context | 86,699 / 183,242 (47.31%) follow destination search; 62,025 (33.85%) follow filter selection | Hotel-ID-only methods have limited signals here | Non-hotel events may still contain informative context |
| Fast immediate continuations | (29,358 + 252,313) / 532,270 = 52.92% have 0–10s gap | Short event intervals characterize many continuations | This gap is measured at target time; not necessarily an online-available feature |
| New-after-exploration clicked hotel often not first candidate | 69,709 / 497,407 (14.01%) in position 1; 159,599 (32.09%) in positions 11+ | Position-only control varies by pattern | Candidate list is stored on target clickout event |
| Histories at cap | 112,648 / 1,276,216 (8.83%) retain exactly 50 events | Potential loss of older behavioral context | Length 50 does **not** establish that any events were discarded |
| Temporal mixture | Immediate: 41.72% train, 41.69% validation, 41.69% test | Stable during this six-day sample | Does not establish seasonal/general population stability |

## Connections to v0.2 and v0.2.1

On the 223,074-example test set, last-item tie-neutral MRR was **0.5049**, recency-300s **0.4944**, and position-only **0.4217**. Last-item's strong aggregate MRR is *consistent with* abundant immediate continuation, not proof that we learned a general traveler preference embedding. The seen/unseen tie-neutral last-item MRR was **0.9092 / 0.1521**. Existing model slices do not yet isolate all four A.2 patterns.

**Do not claim:** that a prior clickout is a booking; that target-row impressions were proven pre-click exposures; that an “unseen” hotel is unseen in the complete session; that 50 retained events always means truncation; or that a GRU will beat current baselines.

## Three unresolved evidence checks

1. **`search for item`:** verify metadata item-ID membership; overlap is supportive, not sufficient for full semantic validation. Sensitivity analysis should consider excluding it from the hotel-action taxonomy.
2. **First versus subsequent raw clickouts:** compare **all raw** clickouts with retained-history categories; emitted examples require at least one previous event and candidate validity.
3. **History length 50:** distinguish exactly-50 complete histories from histories of 51+ preceding events; quantify how many labels were seen only outside retained history.

The read-only `audit_raw_semantics.py` implements these checks. Record its full-data output separately before updating established findings. Raw CSV memory requirements may be substantial.

## Decision gate

Do **not** add context-aware or neural models yet. Reconcile audit counts with the builder and record any discrepancies. Only then choose the next *baseline* question and its fair evaluation design.
