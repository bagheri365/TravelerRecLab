# v0.3.1 — GRU behavioral findings and order perturbation

Status: **post-hoc diagnostic completed; architecture decision pending**.
Inputs: `data/processed/gru_v031_validation_diagnostics.json`, `data/processed/gru_v031_test_diagnostics.json`.
Saved checkpoint: `data/processed/gru_v031.pt` (as recorded in both reports).

## Question and protocol

Does the trained GRU's ranking depend on the recorded order of hotel-ID interactions, and for which behavioral patterns is that dependence useful?

The diagnostic loads a **fixed trained GRU checkpoint** and evaluates each split both on the original history and on one deterministic seeded permutation (seed 42) of the recognized hotel-ID history. The multiset of hotel IDs, candidate set, target, and example population are held fixed. This is **evaluation-time perturbation**, not a shuffled-history-trained GRU and not an isolation of ordering versus recency or model capacity. All numbers are tie-neutral and from the uploaded reports.

## Validation results

| Segment | n | Original MRR | Shuffled MRR | Original − shuffled |
|---|---:|---:|---:|---:|
| All examples | 229,804 | 0.517260 | 0.451011 | +0.066249 |
| Immediate continuation | 95,799 | 0.948313 | 0.763957 | +0.184356 |
| Return to earlier hotel | 11,484 | 0.294383 | 0.478621 | -0.184238 |
| New hotel after exploration | 89,574 | 0.173840 | 0.177424 | -0.003584 |
| No prior hotel reference | 32,947 | 0.275251 | 0.275251 | +0.000000 |

Permutation: 155,141 histories with length >=2; 99,291 histories actually changed; 130,513 unchanged examples.

## Test results

| Segment | n | Original MRR | Shuffled MRR | Original − shuffled |
|---|---:|---:|---:|---:|
| All examples | 223,074 | 0.515058 | 0.449892 | +0.065166 |
| Immediate continuation | 93,005 | 0.947021 | 0.765916 | +0.181104 |
| Return to earlier hotel | 10,940 | 0.290263 | 0.478143 | -0.187880 |
| New hotel after exploration | 86,783 | 0.171776 | 0.174672 | -0.002896 |
| No prior hotel reference | 32,346 | 0.270072 | 0.270072 | +0.000000 |

Permutation: 149,563 histories with length >=2; 94,630 histories actually changed (42.4% of test examples); 128,444 unchanged examples.

Other overall test metrics:

| Metric | Original | Shuffled | Original − shuffled |
|---|---:|---:|---:|
| Recall@5 | 0.603734 | 0.554153 | +0.049581 |
| NDCG@5 | 0.510856 | 0.446187 | +0.064669 |

## Arithmetic decomposition (test)

With mutually exclusive/exhaustive slices, overall MRR change decomposes as

`delta_MRR = sum_g (n_g / N) * (MRR_original,g - MRR_shuffled,g)`.

| Segment | Contribution to overall MRR difference |
|---|---:|
| Immediate continuation | +0.075507 |
| Return to earlier hotel | -0.009214 |
| New hotel after exploration | -0.001127 |
| No prior hotel reference | +0.000000 |

The measured overall change is **+0.065166**. The immediate-continuation segment contributes more than the total positive gain; the earlier-hotel and new-hotel segments offset it. These are aggregate contributions, not estimates of what happened on each actually permuted example.

## Interpretation

1. **Meaningful sensitivity to order:** the fixed GRU performs better on the original histories on both chronological splits (validation delta +0.066249; test delta +0.065166).
2. **Asymmetric sensitivity:** original order is very strong for immediate continuation (test MRR 0.947021), whereas shuffled order is better for returns to earlier hotels (test MRR 0.478143 vs 0.290263). This is *consistent with* excessive preference for the most recent hotel, but that mechanism has not been directly measured.
3. **Limited new-hotel ranking:** original GRU test MRR on new-hotel-after-exploration is 0.171776; ID-only inputs cannot generalize hotel semantics to previously unrecognized candidates. Comparison with position-only MRR 0.299414 is a **different-feature** comparison, not a controlled architecture test.
4. **No-history invariance:** the no-prior-hotel-reference segment has identical metrics under both conditions, as expected if no hotel-ID sequence is available to reorder.

## Limits and next gate

- Only **one random seed / one permutation** was evaluated. Repeated permutations and session-clustered uncertainty estimates are absent.
- Reordering at inference changes the input distribution. This does **not** causally attribute the GRU-versus-mean-pooling advantage to sequence modeling; a matched training-on-shuffled-history control would strengthen that claim.
- Multiple click-outs from the same session are correlated. No uncertainty intervals are supplied.
- Segment definitions use **retained 1–50-event histories**, not complete sessions; event semantics have documented limitations.
- Test slice findings are **post hoc** and must not be used repeatedly for hyperparameter selection.
- The GRU had not plateaued after three epochs; convergence and equal training-budget comparisons remain open.
- Candidate lists are reconstructed from target click-out impressions; availability at serving time is unverified.

**Next step (not implemented here):** on validation, investigate a controlled mechanism to preserve immediate-interest predictions while improving earlier-hotel return and unseen-hotel ranking. Candidate-aware attention over earlier historical states is a possible hypothesis; candidate metadata/context is required for a credible unseen-ID generalization study. Freeze a protocol before testing variants, and reserve test evaluation for the selected model.

## Reproduce

```bash
python -m travelerreclab.models.gru_diagnostics \
  --checkpoint data/processed/gru_v031.pt \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_v031_validation_diagnostics.json \
  --split validation

python -m travelerreclab.models.gru_diagnostics \
  --checkpoint data/processed/gru_v031.pt \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_v031_test_diagnostics.json \
  --split test
```
