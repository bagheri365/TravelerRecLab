# v0.5 — Hotel metadata and cold-start coverage audit (protocol)

**Status:** code ready; full-data measurements **not run**. Do not fill measured values until the JSON report is inspected.

## Question

Can the `item_metadata.csv` property tags cover the candidate hotels that the ID-only GRU-attention encoder struggles to rank, especially new-to-retained-history targets? Do not start implementing a content encoder before checking coverage and distinguishing different kinds of unseen hotels.

## Run

```bash
python -m travelerreclab.data.audit_hotel_metadata_v05 \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/hotel_metadata_audit_v05.json
pytest
```

## Exact definitions

- **Neural train vocabulary membership:** ID appears in *train candidate lists or recognized hotel-bearing train history*, corresponding to the min-count-one vocabulary set (without special PAD/UNK entries). This is a broad presence measure; a hotel need not have been clicked.
- **Train positive-target membership:** hotel appeared as the clicked target in a train example. This is a stricter training-interaction indicator.
- **Target metadata coverage:** clicked hotel ID joins `item_metadata.item_id` (exact string comparison).
- **Target usable properties:** metadata match has at least one nonblank `|`-separated tag.
- **Candidate-slot coverage:** candidate-level counts weighted by how often each candidate appears across logged lists; these are **not independent pre-click exposures**.
- **Behavior:** same four retained-history categories used in v0.3.1/v0.4. No-history means no recognized hotel-bearing event, not necessarily an empty event history.

We report counts and denominators by validation/test and all four behaviors. We also measure distinct relevant metadata tag vocabulary and common tags, each tag counted once per item. No train+validation vocabulary fitting, no use of future click labels in the training vocabulary.

## Evaluation interpretation and limitations

- The metadata table is a static snapshot. We cannot show that its tags were historically available at the event timestamp.
- The target click-out row contains the candidate list; this audit does not establish online availability or causal exposure.
- Exact metadata membership does not prove tag usefulness or predictive value.
- Missing metadata and empty-property metadata are different failure cases.
- Hotel IDs absent from the training vocabulary share the neural UNK representation, even if metadata is present.
- No model training, leakage proof, per-session uncertainty estimation, or architecture choice is claimed here.

## Next decision gate

Only after inspecting actual coverage (especially new-after-exploration, unseen-in-train-vocabulary, and unseen-in-train-target cases) should we design a minimal property-tag candidate encoder and a feature-matched control. Record empirical findings and an ADR then.
