# EDA protocol — mandatory gate before v0.2

## Why EDA now?

The raw dataset has 15,932,992 rows, but this does **not** tell us how many
valid training prefixes, exposed candidates, or independent user histories we
have. EDA determines the prediction unit, feature eligibility, sequence length,
negative sampling, and honest evaluation.

## Observed already (user's v0.0 inspection)

- 15,932,992 events; 730,803 unique users; 910,683 sessions.
- Session lengths: median 4, p90 41, p99 212, maximum 3,522.
- Click-out events: 1,586,586; image interactions: 11,860,750.
- `impressions` and `prices` are 90.04% missing across **all event types**;
  this is not the same as missingness within click-out events.

## Decision-driven questions

| Question | Why it matters | Decision it informs |
|---|---|---|
| What share of click-outs have candidate lists? | Can we evaluate realistic ranking? | Candidate-aware vs catalog retrieval |
| Is target contained in impression list? | Valid ranking examples and data quality | Exclusion and audit policy |
| Are impression and price lists aligned? | Candidate-price correctness | Input validation |
| How many users have multiple sessions? | Is a historical user vector meaningful? | Session-only vs user+session encoder |
| How skewed are click-out steps? | Click-out prefixes may differ from all sessions | 20/50/100 truncation study |
| What is the time range and daily volume? | Detect drift and split boundaries | Chronological splits |
| Are actions sparsely populated by type? | Some fields apply only to specific actions | Typed event encoders |

## Run

```bash
python -m travelerreclab.data.eda_trivago \
  --input data/raw/train.csv \
  --output-dir data/processed/eda \
  --chunksize 250000

cat data/processed/eda/report.md
pytest
```

`summary.json` is machine-readable; `report.md` is a narrative snapshot.
The scanner uses chunks for raw rows but holds user/session aggregate keys in
memory, so available RAM still matters. Reduce `--chunksize` if necessary.
The run may take time on the full 15.9M-event dataset.

## Correct interpretation

- Click-out is a *partner referral*, not a completed hotel booking.
- Candidate lists attached to a click-out record are **not** proven to be
  separately logged pre-click impressions. Treat candidate-aware evaluation
  as an offline approximation, not a fully unbiased logged ranking dataset.
- The `step` distribution is not a verified history-length distribution.
- The script provides descriptive EDA and candidate sanity checks. Before
  claiming a leakage-safe training dataset, separately verify ordering,
  future event exclusion, and temporal fold construction.
- Do not convert provisional design decisions into accepted decisions solely
  because descriptive plots look convincing.

## Interview rehearsal

**Question:** Why didn't you start with a GRU?

**Answer:** "I audited the interaction distribution and click-out candidate
quality first. Dataset semantics determine whether I'm solving next-item
retrieval or candidate ranking; that choice changes labels, negatives, losses,
and metrics. I validate chronological history construction before comparing
sequential architectures."

**Exit gate:** review `summary.json`, confirm candidate coverage and data
quality, and document the target/negative/split decision in ADR-002 and ADR-005.
