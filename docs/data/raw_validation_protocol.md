# v0.1.7 Raw validation protocol

## What this milestone measures

1. **Ordering:** timestamp reversals, tied timestamps, nonincreasing/duplicate steps within a composite user-session key, and session reentry in the original file order.
2. **Emissions:** click-out exclusion reasons classified in the same priority order as `build_sequences.py`: missing target, insufficient history, absent candidates, target outside candidate set, price mismatch. Source-order eligibility is compared to emitted count (1,276,216).
3. **Metadata:** actual `item_metadata.csv` schema, item count, duplicate IDs, missingness, and popular clicked hotel IDs. Popularity can be highly concentrated, so a popularity baseline is required.

## Limitations, consciously retained

- The diagnostic uses *raw CSV source ordering*, whereas the current training builder sorts `(timestamp, step, original order)` per session. A nonzero difference between eligibility and emitted count requires investigation, **not** a speculative explanation.
- A timestamp reversal in raw source order does not prove that builder prefixes leak future events; it indicates ordering deserves attention.
- Reentered session row counts measure rows, not distinct reentered sessions.
- Candidate lists are recorded with click-outs; their independent pre-click availability is unknown.
- Metadata presence and item popularity are measured, but full clicked-item metadata coverage and unseen-item evaluation are **not yet implemented**. Do not claim them.
- The in-memory session maps and per-session step sets are substantial (~910k sessions). Use sufficient memory and report memory constraints. No booking outcomes are present.

## Run

```bash
python -m travelerreclab.data.validate_raw \
  --input data/raw/train.csv \
  --metadata data/raw/item_metadata.csv \
  --output-dir data/processed/raw_validation \
  --chunksize 250000 \
  --emitted-examples 1276216
cat data/processed/raw_validation/report.md
```

## Decision checkpoints

- If duplicate steps/timestamps are common, revise tie-breaking and verify time causality.
- If emitted and source-order eligible counts differ, reconcile by replaying affected sessions in sorted order.
- If cold-start exclusions dominate, add `min_history=0` and a context-only fallback benchmark.
- If hotel popularity is highly skewed, report popularity-stratified ranking performance.
- Audit full clicked-item metadata coverage separately before asserting cold-start generalization.

## Interview drill

**Q: Why audit exclusions instead of training immediately?** The 1.28M examples represent only a subset of the 1.59M click-outs. Ignoring excluded examples could make the benchmark unfairly favorable to history-dependent models.

**Q: Is timestamp sorting enough to eliminate leakage?** No. Prediction timestamps, event availability, target-row impression provenance, and model training cutoffs also matter.

**Q: What does this dataset tell you about embeddings?** It supports action-aware session representations and a click-out proxy, but not direct offline validation of long-term booking intent.
