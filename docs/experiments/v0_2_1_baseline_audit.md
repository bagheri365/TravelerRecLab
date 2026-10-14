# v0.2.1 — Candidate position and tie-neutral baseline audit

**Status:** Implemented, full-data results pending.

## Research questions

1. How much MRR comes from the original candidate-list position?
2. How much of last-item MRR survives neutralizing the inherited candidate-order tie breaker?
3. What are scores for repeat versus non-repeat targets, and short versus long histories?
4. Does validation-only tuning of recency decay change the relative ranking?

## Run

```bash
python -m travelerreclab.evaluation.baseline_audit \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/baselines_v021_audit.json \
  --decays 300 3600 86400
```

For a quick test, add `--max-examples 10000`. The script performs a training-target pass to fit popularity and a held-out pass to evaluate; do not use the test scores to select a decay.

## Metric definitions

`mrr_logged_ties`: score descending, resolving equal scores using the recorded candidate list order.

`mrr_neutral_ties`: expected reciprocal rank if the target's position within its score-tied block is uniform. Similarly for Recall@5 and NDCG@5. For a tied block of size k starting after a strictly-ahead candidates, average over ranks a+1 to a+k. This is deterministic and does not reorder unique scores.

`position`: score `-candidate_index`. It reproduces the recorded order without personalization.

History segments use the emitted (already truncated) prefix. They are overlapping slices: all, target seen/unseen, and 1–5 / 6–20 / 21+ events.

## Results template

| Method | Validation neutral MRR | Test neutral MRR | Test logged MRR |
|---|---:|---:|---:|
| Position | TBD | TBD | TBD |
| Popularity | TBD | TBD | TBD |
| Last item | TBD | TBD | TBD |
| Action weighted | TBD | TBD | TBD |
| Best validation decay | TBD | TBD | TBD |

## Interpretation questions

- Does the last-item baseline still win after tie neutralization?
- Does position-only explain a large part of the original performance?
- Does last-item fail on unseen targets, as expected?
- Are longer histories more useful for aggregation?
- How would we deploy a default when session history is empty (not in this population)?

## Limitations

The clicked hotel's membership in the attached candidate list was verified, but its independent pre-click availability wasn't. This is an offline reconstructed choice task, not a true booking conversion benchmark. Uncertainty intervals and action-weight optimization are not implemented in this milestone.
