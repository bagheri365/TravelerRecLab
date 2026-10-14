# v0.3.1 — GRU results and order-sensitivity diagnostic

**Status:** full first-pass training completed; diagnostic implementation ready; shuffle/slice results pending execution.

## Actual outcomes supplied from the local runs

Full training on 823,338 examples, dim 32, batch 256, three epochs:

| Epoch | Train cross-entropy | Validation tie-neutral MRR |
|---:|---:|---:|
| 1 | 2.654273 | 0.495942 |
| 2 | 2.227473 | 0.512187 |
| 3 | 1.958009 | 0.517260 |

Selected epoch 3; test n=223,074: MRR **0.515058**, Recall@5 **0.603734**, NDCG@5 **0.510856**. Unknown target rate **0.016555**, empty recognized hotel history rate **0.145001**.

Comparators on the same full test population: mean pooling MRR **0.466170**, last-item **0.504871**, position+last-item **0.598007** (position not included in neural inputs). GRU exceeds mean pooling by **0.048889 absolute MRR** and last-item by **0.010188**. These are descriptive point differences, not clustered significance claims.

Smoke run with 5,000 train and 1,000 evaluation examples (2 epochs): GRU validation MRR 0.191559 then 0.212922; test MRR 0.218125, unknown target rate 56.4%. Do not compare smoke quality as if vocabulary/population were identical to the full run.

## Diagnostic protocol

Load `data/processed/gru_v031.pt`, without retraining. On the **same** validation or test examples, evaluate original hotel interaction order and a seeded, per-example randomly permuted order. Preserve recognized hotel-ID multiset, candidate list, label, padding, and the model weights. Report tie-neutral MRR, Recall@5, NDCG@5 and four predeclared behavioral slices, with counts; count sequences actually changed by shuffling. The slices use **original**, retained pre-target histories even in the shuffled condition, so slice membership remains fixed.

Start with validation (no selection/tuning), then run the locked diagnostic on test. The test diagnostic is exploratory/post hoc and **must not** guide architecture or hyperparameter choice. Shuffling can yield identical sequences, particularly with repeated hotel IDs; empty and one-hotel histories are unchanged. An observed drop supports order *sensitivity*, not a causal proof that GRU beats mean pooling because of learned ordering (capacity/optimization differences remain).

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

## Open limitations

Validation MRR was still improving at epoch 3 for both learned encoders. The GRU has more parameters than the pooling model. We have not run repeated seeds, cluster-aware intervals, a retrained order-shuffled control, or a capacity-matched baseline. The candidate set comes from the target click-out row rather than independently verified pre-click exposures. Hotel ID is the only sequence input; true unseen items share UNK. No serving, booking, or causal lift claim is supported.
