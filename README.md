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
