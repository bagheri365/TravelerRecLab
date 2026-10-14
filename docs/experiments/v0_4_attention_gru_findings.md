# v0.4 — Candidate-aware GRU: measured behavioral findings

**Status:** Completed; descriptive offline comparison. **Evidence:** full-population `gru_attention_v04.json`, `gru_attention_v04_validation_diagnostics.json`, `gru_attention_v04_test_diagnostics.json`, and saved v0.3.1 `gru_v031_validation_diagnostics.json` / `gru_v031_test_diagnostics.json`. Report values are transcribed from these artifacts; data and checkpoint files are generated locally and are not bundled in this patch.

## Question and predefined diagnostic

Does candidate-conditioned attention over intermediate GRU states improve `return_to_earlier_hotel` ranking **without sacrificing** `immediate_continuation`? The v0.3.1 GRU represents a session with its final hidden state. v0.4 adds a learned candidate-query attention over prior recurrent states and a gated context with final-state skip; hotel-ID inputs and candidate-softmax training objective remain aligned. The architecture changes **both capacity and scoring**, so attribution to attention alone is unsupported.

**Outcome: hypothesis supported descriptively.** On the held-out test day, return-to-earlier-hotel MRR increases from 0.290263 to 0.359127 (+0.068864, +23.72% relative), while immediate-continuation MRR increases from 0.947021 to 0.949408 (+0.002387). The return segment is 10,940 / 223,074 = 4.90% of test examples.

## Experiment contract and provenance

- Dataset: Trivago RecSys Challenge 2019, filtered next-clickout candidate-ranking examples from `data/processed/next_clickout.jsonl`; 1–50 retained preceding session events. Targets are hotels clicked in the candidate list from their **target clickout rows**, not independently verified pre-click candidate exposure.
- Chronological split: train Nov 1–4, 2018 (823,338); validation Nov 5 (229,804); test Nov 6 (223,074), UTC.
- v0.4 configuration: dimension 32, batch size 256, learning rate 0.003, 3 epochs, seed 42, CPU, train-only hotel vocabulary size 738,063. Best epoch 3 by **validation** tie-neutral MRR. Output `data/processed/gru_attention_v04.pt`.
- Baseline: existing v0.3.1 GRU trained on matching splits and nominal 3-epoch/dimension-32 budget; different parameter counts. Historical GRU results come from its saved diagnostics, not from a newly rerun model.
- Evaluation: tie-neutral expected MRR, Recall@5, and NDCG@5, both aggregate and four **mutually exclusive, retained-history** behavioral segments. Segment taxonomy assigned from original (not shuffled) histories and target.
- The same group counts and hotel-ID vocabulary size appear in v0.3.1 and v0.4 reports, providing a population consistency check; this alone does not prove identical per-example predictions or inferential significance.

## Training and checkpoint selection

| Epoch | Train mean candidate CE | Validation MRR | Validation Recall@5 |
|---:|---:|---:|---:|
| 1 | 2.621944 | 0.501782 | 0.579981 |
| 2 | 2.215438 | 0.517954 | 0.606065 |
| 3 | 1.944437 | 0.522373 | 0.616486 |

Validation MRR kept improving at epoch 3; convergence was not established. The model was selected using validation, *not* segment-level test results.

## Validation results (Nov 5, 2018)

### Tie-neutral MRR

| Segment | Examples | GRU v0.3.1 | Attention-GRU v0.4 | Delta |
|---|---:|---:|---:|---:|
| `immediate_continuation` | 95,799 | 0.948313 | 0.950052 | +0.001739 |
| `return_to_earlier_hotel` | 11,484 | 0.294383 | 0.367959 | +0.073576 |
| `new_hotel_after_exploration` | 89,574 | 0.173840 | 0.173670 | -0.000170 |
| `no_prior_hotel_reference` | 32,947 | 0.275251 | 0.280674 | +0.005423 |
| `all` | 229,804 | 0.517260 | 0.522373 | +0.005113 |

### Recall@5

| Segment | Examples | GRU v0.3.1 | Attention-GRU v0.4 | Delta |
|---|---:|---:|---:|---:|
| `immediate_continuation` | 95,799 | 0.984718 | 0.986721 | +0.002003 |
| `return_to_earlier_hotel` | 11,484 | 0.524391 | 0.684187 | +0.159796 |
| `new_hotel_after_exploration` | 89,574 | 0.285394 | 0.283712 | -0.001683 |
| `no_prior_hotel_reference` | 32,947 | 0.411240 | 0.421092 | +0.009852 |
| `all` | 229,804 | 0.606909 | 0.616486 | +0.009577 |

### NDCG@5

| Segment | Examples | GRU v0.3.1 | Attention-GRU v0.4 | Delta |
|---|---:|---:|---:|---:|
| `immediate_continuation` | 95,799 | 0.956215 | 0.958211 | +0.001995 |
| `return_to_earlier_hotel` | 11,484 | 0.317777 | 0.421574 | +0.103797 |
| `new_hotel_after_exploration` | 89,574 | 0.155141 | 0.154249 | -0.000891 |
| `no_prior_hotel_reference` | 32,947 | 0.268847 | 0.275980 | +0.007133 |
| `all` | 229,804 | 0.513516 | 0.520210 | +0.006694 |

The return-to-earlier-hotel validation MRR improvement is +0.073576, with no immediate-continuation MRR regression. `new_hotel_after_exploration` validation MRR changes slightly negatively; do not claim improvements on every segment across both splits.

## Test results (Nov 6, 2018)

### Tie-neutral MRR

| Segment | Examples | GRU v0.3.1 | Attention-GRU v0.4 | Delta |
|---|---:|---:|---:|---:|
| `immediate_continuation` | 93,005 | 0.947021 | 0.949408 | +0.002387 |
| `return_to_earlier_hotel` | 10,940 | 0.290263 | 0.359127 | +0.068864 |
| `new_hotel_after_exploration` | 86,783 | 0.171776 | 0.172012 | +0.000235 |
| `no_prior_hotel_reference` | 32,346 | 0.270072 | 0.274232 | +0.004160 |
| `all` | 223,074 | 0.515058 | 0.520126 | +0.005067 |

### Recall@5

| Segment | Examples | GRU v0.3.1 | Attention-GRU v0.4 | Delta |
|---|---:|---:|---:|---:|
| `immediate_continuation` | 93,005 | 0.984911 | 0.986666 | +0.001755 |
| `return_to_earlier_hotel` | 10,940 | 0.524625 | 0.678621 | +0.153996 |
| `new_hotel_after_exploration` | 86,783 | 0.279708 | 0.280094 | +0.000386 |
| `no_prior_hotel_reference` | 32,346 | 0.403834 | 0.412520 | +0.008686 |
| `all` | 223,074 | 0.603734 | 0.613428 | +0.009694 |

### NDCG@5

| Segment | Examples | GRU v0.3.1 | Attention-GRU v0.4 | Delta |
|---|---:|---:|---:|---:|
| `immediate_continuation` | 93,005 | 0.955287 | 0.957704 | +0.002417 |
| `return_to_earlier_hotel` | 10,940 | 0.314791 | 0.413376 | +0.098585 |
| `new_hotel_after_exploration` | 86,783 | 0.151837 | 0.151899 | +0.000062 |
| `no_prior_hotel_reference` | 32,346 | 0.262521 | 0.268302 | +0.005781 |
| `all` | 223,074 | 0.510856 | 0.517561 | +0.006705 |

The overall test MRR difference is +0.005067246 (approximately 0.98% relative). On earlier-hotel returns, Recall@5 improves +0.153996 (15.40 percentage points), and NDCG@5 improves +0.098585.

## Why a strong segment gain gives a small overall gain

For disjoint groups g, the arithmetic identity for MRR is:

`delta_MRR_all = sum_g (n_g / N) * (MRR_v04_g - MRR_v031_g)`.

| Segment | n | Share | Segment delta MRR | Weighted contribution to overall MRR |
|---|---:|---:|---:|---:|
| `immediate_continuation` | 93,005 | 41.6924% | +0.002387 | +0.000995 |
| `return_to_earlier_hotel` | 10,940 | 4.9042% | +0.068864 | +0.003377 |
| `new_hotel_after_exploration` | 86,783 | 38.9032% | +0.000235 | +0.000092 |
| `no_prior_hotel_reference` | 32,346 | 14.5001% | +0.004160 | +0.000603 |
| **Total** | **223,074** | **100.00%** | — | **+0.005067** |

The return segment accounts for approximately 66.6% of the aggregate MRR gain, despite representing only ~4.9% of test examples. This is a weighted decomposition, **not** a causal explanation.

## Negative and unresolved findings

1. **New-to-history ranking remains weak.** Test `new_hotel_after_exploration` MRR is 0.172012 (GRU 0.171776); validation decreases slightly from 0.173840 to 0.173670. This extension does not materially resolve unseen-in-history hotel ranking.
2. **Not an attention-only ablation.** Model capacity and the final-state skip also change. A parameter-matched GRU, skip-only baseline, and attention-without-skip comparison would be needed for stronger attribution.
3. **No uncertainty intervals.** Multiple target events per session correlate. We have not computed session-cluster bootstrap CIs or repeated random seeds; improvements are point estimates rather than a statistical significance claim.
4. **No convergence claim.** Both networks' validation MRR improved through epoch three; nominally matched epochs do not guarantee equal training/optimization opportunity.
5. **Hotel-ID cold start.** Hotel IDs outside the training vocabulary share an UNK embedding; no metadata/attributes, action type, time gap, price, search context, or candidate position are used. Test unknown-target rate is 0.016555 and empty recognized hotel-history rate is 0.145001.
6. **Offline candidate-list caveat.** Candidates originate from target clickout rows. This benchmark does not establish booking lift, deployment availability, online latency, or leakage-free exposure collection.
7. **Test evaluation discipline.** Segment results are *post-hoc diagnostics* of the already validation-selected checkpoint; do not choose subsequent architectures or tune parameters against test slices. Future hypotheses should be fixed using train/validation and held for fresh evaluation.

## Reproduction

```bash
python -m travelerreclab.models.gru_attention \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_attention_v04.json \
  --epochs 3 --batch-size 256 --dim 32

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

Use the existing `gru_v031_validation_diagnostics.json` and `gru_v031_test_diagnostics.json` for the baseline group metrics. To verify consistency, require the v0.4 diagnostic aggregate MRR and group sample counts to match the original full-training report and the v0.3.1 segment report respectively.

## Decision-facing conclusion

**Accept v0.4 as an empirically useful (but modest overall) progression of the learned hotel-ID encoder.** Its principal improvement is a large earlier-hotel-return gain, with preserved immediate continuation. Further model complexity is not justified by aggregate MRR alone. A separately designed experiment should target candidate attributes and contextual generalization to new-to-history hotels; no Transformer decision is made here.
