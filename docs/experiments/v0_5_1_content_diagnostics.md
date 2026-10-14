# v0.5.1 — Content-aware GRU: post-hoc behavior and training-coverage diagnostics

## Motivation

The content-aware attention-GRU obtained test tie-neutral MRR **0.5211993** versus the ID-only v0.4 value **0.5201257**, a gain of **0.0010737**. This aggregate improvement does not establish better new-to-history candidate ranking. The original experiment was motivated by the v0.5 metadata audit, which found that most new-to-session targets were *already* in the training vocabulary.

This checkpoint evaluates the **existing validation-selected checkpoint** without retraining, changing its scores, or selecting hyperparameters on test data.

## Slices

The diagnostic reports all examples, four retained-history behavioral classes, and three **mutually exclusive** target coverage classes:

1. `clicked_in_train`: the target hotel was a **positive training click-out**.
2. `in_vocab_not_clicked_in_train`: the target ID appears in training vocabulary (training candidates or hotel history) but was never a positive training target.
3. `outside_train_vocabulary`: the target is absent from the saved training vocabulary and uses the shared hotel-ID `UNK`, though its property tags may differ.

It also reports `with_properties` versus `without_properties` and behavior × training-coverage intersections. All counts describe **target hotels**; they do not independently measure how many candidate negatives are unknown or their relative difficulty. The train-target set is reconstructed from training labels; checkpoint train vocabulary is loaded from the saved model.

The same tie-neutral MRR, Recall@5, and NDCG@5 definitions and target-row candidate lists used in training evaluation are applied to every slice.

## Reproduction

Run validation first:

```bash
python -m travelerreclab.models.gru_attention_content_diagnostics \
  --checkpoint data/processed/gru_attention_content_v051.pt \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051_validation_diagnostics.json \
  --split validation
```

Then use the same protocol on test:

```bash
python -m travelerreclab.models.gru_attention_content_diagnostics \
  --checkpoint data/processed/gru_attention_content_v051.pt \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051_test_diagnostics.json \
  --split test
```

The test `all` slice must reproduce **n=223,074, MRR=0.5211993338, Recall@5=0.6151455021, NDCG@5=0.5188508699** within numerical precision. Validation `all` must reproduce **n=229,804, MRR=0.5233218834**.

Compare behavioral slices with `gru_attention_v04_validation_diagnostics.json` and `gru_attention_v04_test_diagnostics.json`. **Do not compare v0.5.1's unknown-ID slice against the v0.4 aggregate**: to claim cold-start improvement, evaluate the v0.4 checkpoint on the *identical* target-ID coverage slices.

## Prior results, not new findings

| Metric | v0.4 | v0.5.1 |
|---|---:|---:|
| Full test MRR | 0.5201257 | 0.5211993 |
| Full test Recall@5 | 0.6134277 | 0.6151455 |
| Full test NDCG@5 | 0.5175608 | 0.5188509 |

Behavior and target-coverage results are **pending local execution**. In particular, no improvement on unseen-ID or new-to-history examples is claimed yet.

## Limitations

- The content coefficient was fixed at 0.5; encoder parameter counts and content prior differ from v0.4.
- Metadata is a snapshot, not historically point-in-time verified; candidate lists are reconstructed from target click-out rows.
- Hotel train vocabulary includes candidate exposures, not just positive click-outs; this distinction is preserved in the slice definitions.
- Novelty to a traveler, novel training target, and unseen hotel ID are three different populations.
- Slices may be small; one target can occur in multiple correlated click-out examples. No clustered confidence intervals are provided.
- All test slice analyses are descriptive and must not be used for iterative test-driven model selection.
