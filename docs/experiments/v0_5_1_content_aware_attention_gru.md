# v0.5.1 — ID + property candidate encoder in attention-GRU

**Status: protocol and implementation ready; no full-data results yet.**

## Prior evidence and prediction

On the full chronological test split, v0.4 attention-GRU reached tie-neutral MRR
0.520126 overall, but only 0.172012 on the 86,783 `new_hotel_after_exploration`
examples. Our v0.5 audit established that 85,177 of those targets (98.15%)
already occur in the train vocabulary and 86,735 (99.94%) have metadata.
Hence session novelty and global ID cold start are different problems.

**Hypothesis:** hotel property tags can help distinguish candidate hotels not
present in the retained traveler history while preserving direct ID matching.
The hypothesis has not been evaluated yet.

## Model and feature contract

Retain the v0.4 final-state-plus-candidate-attention GRU architecture and
candidate-set cross entropy. Let `e_id(h)` be the train-only ID embedding,
`P(h)` the set of property tags in `item_metadata.csv`, and `v_p` a tag
embedding. For hotel-bearing history events and candidates:

    e(h) = e_id(h) + 0.5 * mean_{p in P(h)} v_p

Absent tags contribute a zero vector. The metadata lookup uses **raw hotel
IDs**, separately from the train-only embedding vocabulary. Thus two unseen
hotel IDs may share the same train-UNK ID embedding and bias but receive
**distinct tag vectors**. A learned tag-only candidate prior also permits
content discrimination when recognized hotel history is empty.

Tag vocabulary comes from item metadata without click-out targets. Hotel-ID
vocabulary comes from training candidates and hotel-bearing history only.
We make no assertion that the metadata snapshot reflects features available
at each historical serving timestamp; this is an offline feasibility study.

The content multiplier `0.5` is an **untuned preset**, not an empirically
selected value. Do not adjust it using test data. Changes in parameter count
and features mean this is not an architecture-only ablation.

## Run

```bash
python -m travelerreclab.models.gru_attention_content \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051_smoke.json \
  --max-train 5000 --max-eval 1000 --epochs 2 --batch-size 128

python -m travelerreclab.models.gru_attention_content \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051.json \
  --epochs 3 --batch-size 256 --dim 32
```

The run writes a JSON metrics report and a `.pt` checkpoint. The checkpoint
stores train-only ID vocabulary and full property tag vocabulary so the model
can be reconstructed; inference also requires the original hotel metadata
snapshot. Training loads metadata into memory; long tag lists increase
batch memory consumption, especially on CPU.

## Evaluation and decision gate

- Same target click-out reconstruction, UTC chronological splits, retained
  history, and candidate-softmax objective as v0.4.
- Select the best epoch on validation **tie-neutral MRR**, then evaluate test
  exactly once. Avoid test-driven hyperparameter/model selection.
- Compare full test MRR, Recall@5, NDCG@5 with v0.4 references:
  `0.520126`, `0.613428`, `0.517561`.
- Evaluate four original behavioral classes using the saved checkpoint.
  New-hotel-after-exploration MRR reference: `0.172012`.
- Within new-to-history examples, report separate slices for (a) target in
  training ID vocabulary, (b) target absent from it, and (c) previously
  positive-labeled training targets. Distinguish these from metadata coverage.
- Check that immediate-continuation MRR (`0.949408`) and earlier-return MRR
  (`0.359127`) do not regress substantially; report regressions even when
  aggregate metrics improve.
- For claims of practical advantage, assess multi-seed variability and
  uncertainty clustered by session; do not claim significance from one run.

## Limitations and deployment implications

Properties mix amenities, accommodation types, and rating-derived tags.
Metadata is not point-in-time verified; impression lists are read from the
**target** click-out event, so candidate availability at serving time remains
unverified. Train ID embeddings are built using exposure candidates as well
as history, not only positive labels. Some cold-start candidates still have
no metadata. This does not yet model search intent, location, price, device,
time gaps, or cross-session traveler state. It is candidate reranking, not
booking prediction, and its offline MRR cannot be translated into online lift.
