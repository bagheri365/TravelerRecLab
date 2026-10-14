# v0.4 — Attention-GRU behavioral segment diagnostic

**Status:** Completed: full chronological validation and test diagnostics available. See [`v0_4_attention_gru_findings.md`](v0_4_attention_gru_findings.md) and [ADR-026](../decisions/ADR-026-attention-gru-behavioral-evidence.md) for measured results and decision.

## Preregistered question

Does candidate-aware attention improve `return_to_earlier_hotel` ranking relative to the v0.3.1 GRU while preserving `immediate_continuation` performance? The full-data v0.4 report recorded 0.520126 test tie-neutral MRR versus 0.515058 for v0.3.1; this alone does not answer the behavioral question.

## Fixed evaluation

Load `gru_attention_v04.pt` (the validation-selected checkpoint from the three-epoch training run). Evaluate the exact same chronological validation/test populations and retained histories used by GRU v0.3.1. Assign behavior groups **from the original hotel-ID history and target**, using the shared v0.3.1 taxonomy. Compute tie-neutral MRR, Recall@5, NDCG@5 by group and overall. Do not retrain or select/tune based on test diagnostics.

```bash
python -m travelerreclab.models.gru_attention_diagnostics \
  --checkpoint data/processed/gru_attention_v04.pt \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_attention_v04_validation_diagnostics.json \
  --split validation

python -m travelerreclab.models.gru_attention_diagnostics \
  --checkpoint data/processed/gru_attention_v04.pt \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_attention_v04_test_diagnostics.json \
  --split test
```

Expected full-population counts: validation 229,804; test 223,074. The overall test MRR should reproduce 0.5201256534402178 within floating-point tolerance; any mismatch needs investigation before segment interpretation.

## Frozen original-GRU test reference

| Behavior | n | v0.3.1 test MRR |
|---|---:|---:|
| immediate_continuation | 93,005 | 0.947021 |
| return_to_earlier_hotel | 10,940 | 0.290263 |
| new_hotel_after_exploration | 86,783 | 0.171776 |
| no_prior_hotel_reference | 32,346 | 0.270072 |
| all | 223,074 | 0.515058 |

## Interpretation limitations

This analysis is post-hoc, point estimates only, and does not isolate the effect of attention from additional capacity or final-state skip design. Session-correlated examples require clustered uncertainty for inferential claims. Hotel-ID-only inputs cannot represent unseen hotel properties semantically; reconstructed clickout-row candidate lists are not independently verified pre-click exposures. Avoid using test slices to select the next architecture.
