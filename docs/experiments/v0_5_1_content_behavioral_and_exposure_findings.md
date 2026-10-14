# v0.5.1 — ID-plus-property attention-GRU: findings and exposure diagnostics

Status: **completed offline exploratory evaluation; results not evidence of production lift**.
Evidence: `data/processed/gru_attention_content_v051.json`, `data/processed/gru_attention_content_v051_{validation,test}_diagnostics.json`, and the corresponding v0.4 ID-only GRU-attention diagnostic JSONs. The latter files are required to reproduce the comparisons; these results have not been inferred from the README.

## 1. Research question and experimental contract

**Hypothesis:** incorporating hotel property tags into the ID-based candidate/history encoding might help rank hotels not yet encountered in the current traveler history. Primary segment: `new_hotel_after_exploration`; secondary: `no_prior_hotel_reference` and training-exposure slices.

- v0.4 comparator: ID-only candidate-aware attention-GRU. v0.5.1: hotel-ID plus property-tag representations, with an additional content-only scoring pathway. The content coefficient is fixed at `0.5`.
- Full-data chronological split: train before 2018-11-05 UTC (823,338 examples), validation on Nov 5 (229,804), test on Nov 6 (223,074).
- Candidate-softmax training; tie-neutral MRR (primary), Recall@5 and NDCG@5 (secondary). Same candidate-ranking benchmark and retained hotel-bearing histories (maximum 50 prior events).
- v0.5.1: embedding dimension 32, batch size 256, learning rate 0.003, seed 42, 3 epochs on CPU, training in file order. `best_epoch=3` selected on validation MRR only.
- 738,063-item training ID vocabulary, 157 property tags (derived from the static metadata file). Hotel IDs unseen in training share ID `UNK` and bias but may differ by tags.
- v0.4 differs in architecture and parameter count, so this is **not** a parameter-count-matched causal isolation of content embeddings. Content encoding and a content-only scoring term both change.

## 2. Learning curves

| Epoch | Train cross entropy | Validation MRR | Validation Recall@5 | Validation NDCG@5 |
|---:|---:|---:|---:|---:|
| 1 | 2.610579 | 0.504026 | 0.584818 | 0.497308 |
| 2 | 2.211296 | 0.519140 | 0.608076 | 0.515374 |
| 3 | 1.944694 | 0.523322 | 0.618283 | 0.521455 |

Training loss decreased and validation MRR increased through epoch 3. **No convergence or optimal training duration has been established.**

## 3. Paired-population model comparison: v0.4 vs v0.5.1

Each row uses the same split and behavioral category for both model results. `delta` means v0.5.1 minus v0.4. All rates below are tie-neutral. Counts are evaluated clickout examples, not independent sessions.

### Validation — MRR, Recall@5, NDCG@5

| Behavior | N | v0.4 MRR | v0.5.1 MRR | Δ MRR | Δ Recall@5 | Δ NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 229,804 | 0.522373 | 0.523322 | 0.000949 | 0.001797 | 0.001244 |
| Immediate continuation | 95,799 | 0.950052 | 0.950694 | 0.000642 | 0.000207 | 0.000541 |
| Return to earlier hotel | 11,484 | 0.367959 | 0.368136 | 0.000177 | 0.003114 | 0.001341 |
| New hotel after exploration | 89,574 | 0.173670 | 0.175063 | 0.001393 | 0.003104 | 0.001932 |
| No prior hotel reference | 32,947 | 0.280674 | 0.281579 | 0.000905 | 0.002407 | 0.001383 |

### Test — MRR, Recall@5, NDCG@5

| Behavior | N | v0.4 MRR | v0.5.1 MRR | Δ MRR | Δ Recall@5 | Δ NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 223,074 | 0.520126 | 0.521199 | 0.001074 | 0.001718 | 0.001290 |
| Immediate continuation | 93,005 | 0.949408 | 0.950185 | 0.000777 | -0.000059 | 0.000548 |
| Return to earlier hotel | 10,940 | 0.359127 | 0.358356 | -0.000771 | -0.001288 | -0.001171 |
| New hotel after exploration | 86,783 | 0.172012 | 0.173175 | 0.001163 | 0.003435 | 0.001877 |
| No prior hotel reference | 32,346 | 0.274232 | 0.276542 | 0.002310 | 0.003235 | 0.002680 |

**Interpretation.** New-hotel MRR improves by a small absolute amount on both splits. Test earlier-hotel-return MRR declines slightly; therefore, there is no across-the-board improvement. The small, consistent aggregate difference is descriptive, without session-clustered confidence intervals.

### Contribution to overall test MRR change

Since the four behavioral groups partition test examples, their weighted differences sum to overall ΔMRR:

\[\Delta\operatorname{MRR}=\sum_g (N_g/N)(\operatorname{MRR}_{v0.5.1,g}-\operatorname{MRR}_{v0.4,g}).\]

| Behavior | Test share | Segment ΔMRR | Contribution to overall ΔMRR |
|---|---:|---:|---:|
| Immediate continuation | 41.692% | 0.000777 | 0.000324 |
| Return to earlier hotel | 4.904% | -0.000771 | -0.000038 |
| New hotel after exploration | 38.903% | 0.001163 | 0.000453 |
| No prior hotel reference | 14.500% | 0.002310 | 0.000335 |
| **Total** | **100%** | — | **0.001074** |

## 4. Training exposure: which target hotels are hardest?

Definitions (mutually exclusive **target** partitions):

- **Clicked in train:** hotel appears as a positive training clickout target.
- **In vocabulary, not clicked in train:** hotel appears in training candidates or hotel-bearing history, but not among positive clickout targets.
- **Outside training vocabulary:** target hotel was absent from the training ID vocabulary; candidate sets may still contain other unknown IDs.

These are *not* the same distinction as `new_hotel_after_exploration`: the latter means absent from the **traveler's retained session history**, not unseen in global training.

### Full-split exposure categories — v0.5.1 only

| Split | Target exposure | N | MRR | Recall@5 | NDCG@5 |
|---|---|---:|---:|---:|---:|
| validation | clicked_in_train | 195,134 | 0.548107 | 0.645659 | 0.547833 |
| validation | in_vocab_not_clicked_in_train | 31,011 | 0.396758 | 0.469317 | 0.384490 |
| validation | outside_train_vocabulary | 3,659 | 0.274219 | 0.420849 | 0.275521 |
| test | clicked_in_train | 188,620 | 0.546234 | 0.642763 | 0.545485 |
| test | in_vocab_not_clicked_in_train | 30,761 | 0.397198 | 0.469816 | 0.384868 |
| test | outside_train_vocabulary | 3,693 | 0.275426 | 0.415101 | 0.274539 |

### New-to-history selection crossed with training exposure

| Split | New-hotel target exposure | N | MRR | Recall@5 | NDCG@5 |
|---|---|---:|---:|---:|---:|
| validation | clicked_in_train | 74,156 | 0.184689 | 0.311910 | 0.169581 |
| validation | in_vocab_not_clicked_in_train | 13,811 | 0.126506 | 0.162407 | 0.088830 |
| validation | outside_train_vocabulary | 1,607 | 0.148168 | 0.198002 | 0.116717 |
| test | clicked_in_train | 71,467 | 0.183486 | 0.309430 | 0.167909 |
| test | in_vocab_not_clicked_in_train | 13,710 | 0.121731 | 0.157112 | 0.083606 |
| test | outside_train_vocabulary | 1,606 | 0.153476 | 0.210176 | 0.123917 |

**Caution:** performance varies by exposure group, but the groups differ in prevalence, historical click propensity, competition in candidate lists, and other confounders. Out-of-vocabulary MRR exceeding in-vocabulary-never-clicked MRR on the new-hotel test slice is **not** proof of superior cold-start behavior. We do not have ID-only v0.4 scores on those identical exposure groups, so **the cold-start improvement hypothesis has not been tested comparatively**.

## 5. Property coverage, limits, and data contract

- Validation: with property tags **229,701**, without tags **103** targets; absence of metadata is rare but nonzero.
- Test: with property tags **222,984**, without tags **90** targets; absence of metadata is rare but nonzero.

- Train-vocabulary coverage is determined from the saved checkpoint; positive train-target membership is derived only from training clickouts.
- Metadata is a **static snapshot**, without verification that each attribute (especially rating-derived tags) existed at the historical prediction timestamp. Point-in-time leakage remains unresolved.
- Candidates come from the **target clickout row**, not from independently audited pre-click exposures. This is a reconstructed candidate-ranking benchmark, not proof of live serving feasibility.
- History is limited to hotel-bearing actions retained in the preceding 50 events. The model does not use action type, time gap, filters, destination, prices, or candidate position.
- Multiple targets can belong to the same session; there are **no session-clustered uncertainty estimates** or controlled repeated-seed training runs.
- Three epochs, one random seed, file-order batching, and differing model capacity limit architectural comparisons.
- The evaluations are post hoc, descriptive diagnostics. **Do not tune additional models on the test breakdown.**
- Offline clickout MRR is not booking rate, conversion lift, or evidence of a reliable production representation.

## 6. Research judgment and follow-up gate

**Judgment:** the content-aware extension shows a **small positive aggregate effect** and a small improvement on new-to-history selection, but it does not materially resolve new-hotel ranking. The original primary hypothesis is therefore **weakly supported**, and **genuine ID cold-start benefit remains unestablished**. The small return-to-earlier-hotel regression must be retained in the record.

Before a new model: (1) verify context-field event-time semantics/availability, (2) define a leakage-safe search-intent representation (destination, filters and relevant context), (3) keep both behavior and exposure slices in validation reporting, and (4) compare only on held-out test after selecting the new procedure on validation. An alternative pure content ablation, repeated seeds, or session-clustered uncertainty can be scheduled as a separate preregistered investigation.

## 7. Reproduction

```bash
python -m travelerreclab.models.gru_attention_content \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051.json \
  --epochs 3 --batch-size 256 --dim 32

python -m travelerreclab.models.gru_attention_content_diagnostics \
  --checkpoint data/processed/gru_attention_content_v051.pt \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051_validation_diagnostics.json \
  --split validation

python -m travelerreclab.models.gru_attention_content_diagnostics \
  --checkpoint data/processed/gru_attention_content_v051.pt \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051_test_diagnostics.json \
  --split test
```

The v0.4 comparison additionally needs `gru_attention_v04_{validation,test}_diagnostics.json`; do not infer these per-segment reference metrics from aggregate logs.
