# v0.1.6 — Targeted sequence EDA

This diagnostic audits emitted JSONL examples from the v0.1 generator. It does
**not** scan the raw source for event-order defects or excluded click-outs.

## Run

```bash
python -m travelerreclab.data.build_sequences \
  --input data/raw/train.csv \
  --output data/processed/next_clickout.jsonl \
  --max-history 50 --min-history 1

python -m travelerreclab.data.eda_sequences \
  --input data/processed/next_clickout.jsonl \
  --output-dir data/processed/sequence_eda \
  --split-day 2018-11-05

cat data/processed/sequence_eda/report.md
pytest
```

## Questions and decision use

| Measurement | What it informs | Caveat |
|---|---|---|
| Emitted prefix-length p50/p90/p99 | Truncation sensitivity | Already capped at 50 |
| Same-item target vs history | Last-item/repeat baselines | Only item-bearing action types |
| Emitted-target candidate coverage | Data-pipeline sanity | Rows failing candidate validation were excluded |
| Time-fold target counts | Temporal evaluation design | Fold boundary is a hypothesis |
| User/session overlap across folds | Identity-awareness | Overlap alone is not target leakage |
| Order checks on emitted histories | Regression guardrail | Generator has already sorted the data |

## Before v0.2: limitations to resolve

1. Compute retained/excluded target rate against **all 1,586,586 raw click-outs**.
2. Audit raw `(user_id,session_id,step)` uniqueness, monotonic timestamp vs
   step, tied timestamps, and input order, not only generated JSONL.
3. Explicitly select and freeze the time-split and negative/candidate protocol.
4. Confirm whether the next-click-out target should be scored among
   recorded target-row candidates or with catalog-wide negative sampling.
5. Keep a written explanation of the non-pre-click nature of `impressions`.

## Interview discussion

**Why not immediately train a GRU?** Because sequence length, historical item
repetition, target retention, and evaluation eligibility determine whether a
GRU's complexity is valuable. EDA defines the experimental unit and guards
against unjustified causal and serving claims.
