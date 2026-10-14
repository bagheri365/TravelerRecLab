# v0.2.3 — Position + last-hotel controlled baseline

**Status: protocol implemented; full-data results pending.** This checkpoint adds **one** candidate-ranking hybrid, not a new representation learner.

## Question and hypothesis

Can a minimal combination of the logged candidate order and the latest hotel-specific reference outperform either simple signal by itself? Prior test results: position-only tie-neutral MRR **0.4217**, last-item **0.5049**. On target-unseen examples, position-only scored **0.3421** and last-item **0.1521**; on seen targets the order reversed (last-item **0.9092**, position **0.5129**). These figures come from v0.2.1 and are not results of this new experiment.

## Fixed evaluation contract

- Unit: v0.1 history-eligible next-click-out example, candidate list from the target click-out row. This list is **not a separately confirmed pre-click exposure**.
- UTC time split: train before Nov 5, validation Nov 5, test Nov 6, 2018. This experiment learns **no** training-set parameters; validation selects the boost.
- The historical reference must be from a hotel-typed action, including provisional `search for item`. Destination/filters/POIs are excluded.
- Scores per candidate at zero-based position `i` in a list of `n`:

  `position(i) = 1 - i/(n-1)` for `n>1` (singleton: 1).

  `hybrid(h) = position(h) + boost * 1[h = last_history_hotel]`.

  `boost` searched over `[0, 0.25, 0.5, 1, 2, 4]`. `boost=0` is the position-only control. This normalization keeps boost comparable across candidate counts.
- Primary metric: **tie-neutral MRR**, exactly as defined in v0.2.1. Secondary metrics: tie-neutral Recall@5 and NDCG@5; original-order variants remain diagnostics.
- Select **one** boost maximizing validation **all-example** tie-neutral MRR, taking the smallest boost on ties. View test scores only after validation selection; avoid retrospectively changing the search grid.
- Report the four A.2 *mutually exclusive retained-history patterns*: immediate continuation, earlier-hotel return, new after exploration, and no prior hotel reference. These are descriptive slices, not independent tuning targets.

## Run commands

Smoke test (first 10,000 eligible examples per split, **not representative**):

```bash
python -m travelerreclab.evaluation.hybrid_position \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/hybrid_position_v023_smoke.json \
  --max-examples 10000
```

Full experiment:

```bash
python -m travelerreclab.evaluation.hybrid_position \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/hybrid_position_v023.json
```

## Results (fill only after execution)

| Model | Validation neutral MRR | Test neutral MRR | Test Recall@5 |
|---|---:|---:|---:|
| Position-only (boost 0) | TBD | TBD | TBD |
| Validation-selected boost | TBD | TBD | TBD |
| Last-item (v0.2.1 reference) | 0.5049 | 0.5049 | TBD |

For each of the four patterns, record `n`, test tie-neutral MRR, and Recall@5 for boost=0 and the selected hybrid. Note that metrics within a pattern such as immediate continuation will be artificially favorable to an oracle-like match rule but the model itself only sees history, not future target labels.

## Interpretation safeguards

1. **Position is a property of the upstream ranker**, not evidence that a traveler embedding improved. A successful hybrid is a better *candidate reranker* under this offline protocol.
2. Candidate lists on target click-out rows are retrospective observations, not guaranteed available at prediction time. Production feasibility requires a serving-time candidate list and documented provenance.
3. The hybrid does not handle hotel similarity or destination search intent. In no-hotel-reference histories it reduces to position-only.
4. The history window is limited to 50 events. Full-session and zero-history populations are out of scope.
5. Examples within a session are correlated; this checkpoint does not calculate clustered uncertainty intervals or booking lift.

## Decision gate

Compare selected hybrid with position-only on overall tie-neutral MRR and A.2 patterns. Preserve last-item as the behavioral-only control. Document whether improvement is concentrated on immediate continuation and whether it hurts exploration. **Do not introduce metadata models or sequence models from this experiment alone.**
