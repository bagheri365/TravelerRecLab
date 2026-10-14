# TravelerRecLab

Real-time traveler representations for hotel recommendation: compare recency-weighted aggregation with GRU and Transformer sequence encoders, then test online state updates.

## Scope and milestones

- **v0.0 (this patch):** package scaffold, Trivago event-log audit, synthetic fixture tests.
- **v0.1:** temporal ordering, session prefixes, and leakage-safe next-click-out labels.
- **v0.2–v0.3:** popularity, mean, action-aware, and recency-weighted baselines.
- **v0.4–v0.6:** GRU4Rec, time-aware GRU, and causal Transformer.
- **v0.7–v0.8:** retrieval losses and intent-shift ablations.
- **v0.9–v1.0:** deterministic online GRU replay, state versioning, latency study.

## Install (Python 3.12)

```bash
python -m pip install -e '.[dev]'
pytest
```

## Data

Primary: [Trivago RecSys Challenge 2019](https://www.recsyschallenge.com/2019/). Obtain data according to the challenge terms and place `train.csv` in `data/raw/`. Raw event logs are **not** committed. It includes user/session event sequences and click-outs, **not confirmed bookings**.

Expected fields checked by the audit: `user_id`, `session_id`, `timestamp`, `step`, `action_type`, `reference`, `impressions`, `prices`.

Run:

```bash
python -m travelerreclab.data.inspect_trivago --path data/raw/train.csv
```

The audit reports shape, event types, missingness, user/session counts, and session-length distribution; no model training occurs in v0.0.

### Other data (separate experiments)

- [Expedia Hotel Recommendations](https://www.kaggle.com/competitions/expedia-hotel-recommendations/data): hotel-cluster click/booking events; cannot be identity-joined with Trivago.
- [Expedia Personalized Sort](https://www.kaggle.com/competitions/expedia-personalized-sort/data): impression-level click/booking ranking data.

Data licenses and download access are separate from this repository's source-code license.

## v0.1 — Causal next-click-out dataset

After downloading Trivago `train.csv`, run:

```bash
python -m travelerreclab.data.build_sequences \
  --input data/raw/train.csv \
  --output data/processed/next_clickout.jsonl \
  --max-history 50 --min-history 1
```

Each JSONL row contains the **strictly previous** session events as `history`,
`target_item` (the next click-out), plus the candidates/prices logged on that
click-out event. Sorting is `(user_id, session_id, timestamp, step, input row)`.
Sessions use the composite key `(user_id, session_id)`.

**Important evaluation limitation:** Trivago's click-out-row impression list is
not independently timestamped as an earlier exposure event. Treat it as an
*offline candidate set* for evaluating next click-out prediction, not proof that
these candidates were known at an earlier prediction timestamp. The target
click-out is excluded from the encoded history. This milestone does **not**
create booking labels or claim unbiased impression-level click prediction.
Rows with missing target items, invalid candidate membership, or mismatched
candidate/price lengths are skipped. The v0.1 pipeline loads the CSV in memory;
chunked processing is a future scalability improvement.

## Design rationale and interview preparation

This repository records choices as explicit [architecture decision records](docs/decisions/README.md), not just implementation notes. Read the [observed Trivago data audit](docs/data/trivago_audit_2026-10-13.md), use the [experiment template](docs/experiments/experiment_template.md) to record evidence, and rehearse the [interview decision drill](docs/interview/decision_drill.md).

**Evidence status:** v0.1 and model defaults are not automatically validated by a successful test run. Next-click-out is a proxy, not a completed booking; target-row candidate lists are an offline evaluation construct, not verified past exposure.

## v0.1.6 — Targeted sequence and leakage EDA

After generating `data/processed/next_clickout.jsonl`, run:

```bash
python -m travelerreclab.data.eda_sequences \
  --input data/processed/next_clickout.jsonl \
  --output-dir data/processed/sequence_eda
```

Read `data/processed/sequence_eda/report.md` and
[the diagnostic protocol](docs/data/sequence_eda_protocol.md). Decision
rationale: [ADR-008](docs/decisions/ADR-008-eda-driven-modeling-scope.md).
The diagnostic inspects **emitted examples**, not all raw click-outs; it cannot
certify source-event ordering or point-in-time exposure availability.

## v0.1.7 — Raw data validation before baselines

Audit ordering anomalies, click-out exclusion reasons, and hotel metadata/popularity:

```bash
python -m travelerreclab.data.validate_raw \
  --input data/raw/train.csv \
  --metadata data/raw/item_metadata.csv \
  --output-dir data/processed/raw_validation \
  --emitted-examples 1276216
```

See [validation protocol](docs/data/raw_validation_protocol.md) and [ADR-009](docs/decisions/ADR-009-raw-validation-before-baselines.md). Results must be interpreted as *source-order* diagnostics; the v0.1 sequence builder sorts events.
