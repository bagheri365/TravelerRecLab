# v0.1.8 — Final EDA protocol

## Why this milestone exists

The raw audit reported **1,276,213** source-order eligible click-outs but the sequence generator emitted **1,276,216** timestamp/step-sorted examples. We should **explain**, not conceal, the difference before benchmarking models.

## Reconciliation method

The current generator uses `min_history=1`. Apart from candidate checks, a click-out fails the history requirement only when it is the **first event of its session**. For each `(user_id, session_id)`, compare the source-first event with the event first under `(timestamp, step, original row position)` ordering. A valid click-out that moves away from first place gains eligibility; one that moves to first place loses eligibility. Sum those changes to predict the emitted-minus-source-order difference.

This identity does **not** hold without modification for `min_history > 1` or for extra filtering rules. It reconciles counts; it does not prove timestamp-first ordering is semantically correct. Inspect sample affected sessions and record the eventual event-ordering contract separately.

## Metadata & popularity

Count all clicked hotel IDs, then join membership against `item_metadata.csv`. Report both distinct-item coverage and click-out-weighted coverage because popularity makes them differ. Count metadata property tags split on `|`; examine item popularity concentration. These are descriptive analyses, not results of predictive models.

## What this does not prove

- Candidate impression lists on click-out rows were available at ranking-request time.
- Hotel properties were always available historically.
- Hotel IDs seen as `reference` under non-hotel actions are hotel identifiers.
- Six days of Trivago sessions capture long-term traveler preference changes.

## How to run

```bash
python -m travelerreclab.data.audit_final \
  --input data/raw/train.csv \
  --metadata data/raw/item_metadata.csv \
  --output-dir data/processed/final_eda \
  --chunksize 250000 \
  --expected-delta 3
cat data/processed/final_eda/report.md
```

## Interview drill

**Question:** Why did two eligibility counts disagree by three examples?

**Answer:** Because the raw-order audit and sequence generator ordered events differently. When requiring one prior event, only the identity of the first event determines history exclusion; I reconciled the two methods by comparing each session's first event, then documented any differences rather than ignoring a small discrepancy.

**Question:** Why measure both item-weighted and interaction-weighted metadata coverage?

**Answer:** If popular hotels have better metadata coverage, an interaction-weighted number may look excellent while many rare/cold items remain unsupported. Both views matter for robust hotel embeddings.
