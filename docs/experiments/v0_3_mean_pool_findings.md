# v0.3 findings — Learned hotel-ID mean pooling

**Status:** Full experiment run reported by project operator; interpretation checkpoint. **No model change in this document.**

## Research question and scope

Can a trainable, non-sequential hotel-ID representation improve next-click-out ranking compared with fixed behavioral heuristics? This experiment predicts the clicked hotel within the candidate list attached to its click-out record. It does **not** predict bookings, evaluate complete-catalog retrieval, or verify pre-click candidate exposure. Histories contain at most 50 preceding session events; only hotel-referencing actions enter the encoder. The model excludes candidate position, prices, amenities, destination, device, action-type embeddings, and timing features.

Architecture: trainable hotel-ID embeddings; mean of eligible historical hotel embeddings (learned empty-history vector when none); dot-product against candidate hotel embeddings, scaled by square root of dimension, plus candidate bias; candidate-set softmax cross-entropy. IDs absent from the **training vocabulary** share an UNK embedding. Training vocabulary includes training candidate IDs and hotel histories, not just clicked targets.

Chronological splits: training before 2018-11-05, validation 2018-11-05, test 2018-11-06 onward. Checkpoint choice uses validation MRR; test is an evaluation, not a tuning set.

## Reproduction commands

```bash
python -m travelerreclab.models.mean_pool \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/mean_pool_v03_smoke.json \
  --max-train 5000 --max-eval 1000 --epochs 2 --batch-size 128

python -m travelerreclab.models.mean_pool \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/mean_pool_v03.json \
  --epochs 3 --batch-size 256 --dim 32
```

## Recorded results (from terminal output, not independently rerun here)

### Smoke run

| Epoch | Train examples | Training loss | Validation tie-neutral MRR |
|---|---:|---:|---:|
| 1 | 5,000 | 2.971050 | 0.300062 |
| 2 | 5,000 | 2.856505 | 0.300661 |

Test (n=1,000): **MRR 0.289948**, Recall@5 **0.370528**, NDCG@5 **0.270757**. Unknown target rate **56.4%**, empty hotel-history rate **15.7%**. First-N sampling by chronological split is a *wiring test*, not a representative training-size experiment.

### Full training run

| Epoch | Train examples | Training loss | Validation tie-neutral MRR |
|---|---:|---:|---:|
| 1 | 823,338 | 2.892988 | 0.445310 |
| 2 | 823,338 | 2.466521 | 0.459424 |
| 3 | 823,338 | 2.174435 | 0.467356 |

Test on **223,074** examples: **tie-neutral MRR 0.466170**, **Recall@5 0.571227**, **NDCG@5 0.463836**. Unknown target rate **1.6555%**, empty hotel-history rate **14.5001%**.

The training loss decreased and validation MRR increased at every observed epoch. The best evaluated epoch was 3, the final epoch; this is **not evidence of convergence** or an optimal epoch budget. The smoke and full runs differ in training size, vocabulary, epochs and other conditions, so their performance difference has no single established cause.

## Comparison with completed full-test baselines

All values below are previously reported tie-neutral MRR on the **223,074-example test split**. The comparison is descriptive; the hybrid consumes a different feature source.

| Model | Tie-neutral test MRR | Relevant information |
|---|---:|---|
| Popularity | 0.259990 | Training-period target counts |
| Position-only | 0.421720 | Original target-row candidate order |
| **Learned mean pooling** | **0.466170** | Trainable hotel-ID history and candidate embeddings + biases |
| Action-weighted history | 0.477068 | Fixed action-weighted hotel IDs |
| Recency 300 seconds | 0.494395 | Fixed hotel-ID recency scoring |
| Last-item | 0.504871 | Most recent hotel identity |
| Position + last-item, boost 2 | 0.598007 | Original candidate position + latest hotel |

The learned mean-pooled model beats popularity, but does not beat the fixed action-weighted, recency-weighted, or last-item baselines on MRR. It should **not** be described as an apples-to-apples comparison with position-only or the position-plus-history hybrid, because those methods use candidate-order information unavailable to the neural model.

## What we know versus what remains unproven

**Established from these runs:** training proceeded on 823,338 examples; the objective decreased over three epochs; validation MRR improved to 0.467356; full-test MRR was 0.466170; 1.6555% of test target IDs were outside the train-built vocabulary; 14.5001% of test examples had no recognized hotel history.

**Plausible, not established:** Equal mean weighting underemphasizes the latest hotel and helps explain the last-item advantage. This must be tested with controlled encoders or segment results; optimization time, frequency effects, candidate bias, and missing context are also possible explanations.

**Not measured in this run:** Four-pattern segmentation; per-segment model MRR; train-vocabulary size; validation unknown rate; per-epoch wall time; convergence; random-seed variability; per-session confidence intervals; distinction between unknown target IDs and unknown candidate IDs; effect of session-order shuffling; metadata-based cold start.

**Important nuance:** Unknown target rate is a vocabulary-coverage metric. Because the vocabulary includes *training candidates*, it is not interchangeable with "hotel never clicked in training" or "hotel never seen in session history."

## Risks and experimental decisions

1. **No GRU superiority claim yet.** The model is a non-sequential learned control, not a GRU comparison.
2. **Do not claim full convergence.** Validation scores are still improving at the last tested epoch. A subsequent comparison should allow matched, validation-selected training budgets (or extend the mean-pooling run using validation only).
3. **Hold features fixed in architectural ablations.** GRU versus mean pooling should use the same ID vocabulary, inputs, candidate set, scorer/loss, split, and evaluation metrics; if action/time features are introduced, give them to both.
4. **Retain separate ranking baselines.** Position-augmented hybrid remains an external reranking reference, not a history-only neural comparator.
5. **Do not infer cold-start capability.** A single UNK vector cannot distinguish different unseen hotel IDs.
6. **Report by behavior.** Four categories (immediate continuation, return to earlier hotel, new after exploration, no prior hotel reference) are essential to interpret aggregate lift. The current CLI's output does not provide those segment metrics.

## Interview explanation (short)

> I trained an ID-embedding model that mean-pools earlier hotel interactions and uses candidate-softmax to predict the next click-out. On 223,074 chronological test examples it achieved 0.466 tie-neutral MRR, while a last-item baseline reached 0.505. That shows the trainable representation is viable but does not prove it captures short-term intent better than recency heuristics. Its best validation checkpoint was the final tested epoch, so I would avoid claims of convergence and compare it to a feature-matched GRU under validation-selected training budgets. I would also inspect the four behavioral segments before deciding whether order modeling adds value.

## Next checkpoint (not yet implemented)

Review the full `mean_pool_v03.json` and checkpoint metadata for selected epoch, seed, vocabulary size, and environment; then decide whether a longer mean-pool reference run and a common segmented evaluator are required before GRU training. No test-set-driven hyperparameter changes.
