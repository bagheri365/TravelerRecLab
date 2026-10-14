# v0.2 — Non-sequential candidate-ranking baselines

**Status:** Implemented, not yet evaluated on full data. Fill results only after running the command.

## Why this experiment?

The sequence EDA found that about 46.7% of emitted click-out targets were previously referenced in their session. We need to determine whether simple repeated-interest rules beat global popularity before introducing GRUs/Transformers.

## Task and population

One row = a click-out with >=1 prior session event and a valid candidate list. Predict the clicked hotel among the *recorded click-out-row* candidate items. This is an **offline reconstruction**, not a validated pre-click ranking impression experiment. Completed bookings are not observed. The current builder excludes zero-history click-outs; report this coverage restriction alongside model metrics.

## Split (UTC target timestamp)

- Train: before 2018-11-05 (Nov 1–4 in this extract)
- Validation: 2018-11-05
- Test: 2018-11-06 onward (Nov 6 in this extract)

Global popularity counts are fit ONLY on eligible training-target records. No validation or test targets enter the popularity table. Session histories in validation/test may legally contain earlier observations, including earlier periods. No hyperparameter may be chosen by inspecting test metrics.

## Baselines and scoring contracts

- `random`: deterministic SHA256-based per-candidate random score; sanity reference, not expected to be exactly 1/25 on every list.
- `popularity`: count prior *training-target* hotel click-outs; ties preserve recorded candidate order.
- `last_item`: last hotel-typed reference in session history. Destination/POI/filter references excluded.
- `action_weighted`: sum fixed action-specific weights for history references matching a candidate.
- `recency_weighted`: action-weighted sum with exp(-age_seconds / tau). Initial tau = 3,600 seconds, provisional.

All heuristic scores use the same candidate lists and deterministic ties (recorded candidate order). When history contains no hotel-typed events, candidate positions determine ranking; **this is an important potential source of list-position bias** and should be compared to position-only baseline later.

Current hotel-typed actions are image/info/rating/deals interactions, click-outs, and search-for-item. Verify that `search for item` references actually represent hotel IDs in this dataset; exclude it if its reference semantics differ.

## Metrics

Single positive target per candidate set: MRR, Recall@5, Recall@10, NDCG@5, NDCG@10. The candidate set has typically 25 items. No AUC, booking conversion, or online results can be inferred from these metrics.

## Run

```bash
python -m travelerreclab.evaluation.baselines \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/baselines_v02.json
```

For quick correctness checks use `--max-examples 10000`, which limits examples PER holdout split, not training popularity fitting.

## Decision log

| Question | Current choice | Alternative | Revisit if |
|---|---|---|---|
| Is order essential? | Compare heuristics before GRU | Train GRU immediately | Sequential model beats tuned non-sequential baseline |
| Is history enough? | Use short-term session history | Multi-session user embedding | Long-term lift appears in cohort analyses |
| Can we use click-out candidate lists? | Offline candidate-set benchmark | Independently logged impressions | Exposure timing confirmed |
| How to split? | Chronological by target time | Random prefix split | More representative full-length data available |
| Why 1h decay? | Initial interpretable hyperparameter | Tune tau on validation | Validation suggests another decay rate |

## Results (to fill after full run)

| Model | Validation MRR | Test MRR | Test Recall@5 | Test NDCG@5 |
|---|---:|---:|---:|---:|
| Random | TBD | TBD | TBD | TBD |
| Popularity | TBD | TBD | TBD | TBD |
| Last item | TBD | TBD | TBD | TBD |
| Action weighted | TBD | TBD | TBD | TBD |
| Recency weighted | TBD | TBD | TBD | TBD |

## Interview explanation

> I fixed a chronological ranking protocol first, then tested random, popularity, last-property, action-weighted and recency-weighted baselines using identical candidate sets. All population popularity counts were fitted on the training period only. I distinguished candidate-list reconstruction from true pre-click exposure and explicitly reported that zero-history decisions are excluded. That establishes whether sequence-model complexity is justified.
