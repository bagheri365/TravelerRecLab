# v0.3 — First learned traveler representation: hotel-ID mean pooling

**Status: implementation ready, full-data results not measured.** This is our first *trained* representation model, not yet a GRU. The next controlled experiment can replace the mean aggregator with a GRU using the *same* embeddings, candidate softmax, splits and evaluation. No position features are included; compare directly with behavior-only baselines, not with the position-augmented hybrid as though input information were equal.

## Question and hypothesis
Can a learned vector average of past hotel interactions distinguish next click-out candidates beyond fixed item-ID heuristics? Hypothesis: trainable hotel embeddings capture co-interaction structure, but cold item IDs and mixed short-term intent limit gains. A negative result is informative.

## Task and input contract
Each eligible click-out example has `history` of up to 50 preceding events, `candidate_items` from the target click-out row, `target_item`, and `target_timestamp`. Hotel-bearing event references only (hotel image/info/rating/deals, clickout, search-for-item) enter the history encoder. Other actions are intentionally excluded in this first ablation. This is a conditional candidate-set benchmark, **not** verified pre-click exposure data. No booking labels. History-less raw clickouts excluded.

## Architecture and loss
Trainable item-ID embedding table `E ∈ R^{V×d}`, optional candidate item bias `b_h`, with padding and unknown tokens. For hotel IDs from history, `z = mean_i E[h_i]`; empty hotel history uses a learned empty vector. Candidate logits `s_j = (z · E[c_j]) / sqrt(d) + b_{c_j}`. Candidate-set softmax cross-entropy minimizes `-log exp(s_target)/sum_j exp(s_j)`. Candidates are **the logged list**; we do not generate random negatives. Train on clicked target labels only. `UNK` maps IDs unseen in *train* to one shared vector; this is a limitation, not hotel cold-start handling. No price/context/position.

## Leakage and split rules
Train before 2018-11-05; validate 2018-11-05; test 2018-11-06 onward, target UTC. Vocabulary built from **train examples only**, including train candidate items and hotel history, avoiding validation/test vocabulary fitting. Training uses train targets only. Validate each epoch; choose highest validation tie-neutral MRR, earliest epoch on ties; evaluate test once after selection. For fair claims, do not alter architecture based on observed test results.

## Installation and run

PyTorch is deliberately optional so existing project EDA dependencies stay lightweight. In the activated environment: `python -m pip install torch`.

Quick wiring smoke run (first N examples *within each chronological split*; not representative):

```bash
python -m travelerreclab.models.mean_pool \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/mean_pool_v03_smoke.json \
  --max-train 5000 --max-eval 1000 --epochs 2 --batch-size 128
```

Full reference run (CPU can be slow; vocabulary includes train candidate IDs):

```bash
python -m travelerreclab.models.mean_pool \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/mean_pool_v03.json \
  --epochs 3 --batch-size 256 --dim 32
```

Generated JSON contains epoch trace, selected epoch, tie-neutral test MRR/Recall@5/NDCG@5, unknown-target rate, and artifact path. `.pt` checkpoint contains fitted vocabulary and model state. These files are experimental outputs and should not be silently committed or distributed without reviewing size and provenance.

## Practical limitations and alternatives
- Sequential order within the retained hotel references is not modeled. This is deliberate: it establishes the *learned non-sequential* control for a future GRU.
- No action-type or timing embeddings; they could be introduced as a feature-matched extension to *both* mean pooling and GRU.
- Training streams JSONL each epoch but builds a vocabulary in an initial pass; large train candidate vocabularies can demand memory.
- Train examples follow JSONL order (no global shuffle); training dynamics can depend on file ordering.
- Tied unknown scores are evaluated by expected random tie ordering; training softmax sees tied unknown candidate logits.
- Candidate bias may learn candidate popularity; this helps the benchmark but complicates interpreting the traveler vector alone. Compare against popularity baseline.
- Per-session correlated examples, no clustered confidence intervals yet. Six days only.
- Trivago does not supply booking outcomes, and there is no independently verified pre-click exposure logging.

## Decision record / report template
Record dataset fingerprint, command, seed, environment/PyTorch version, train count, vocabulary size, unknown rate, epoch-wise losses and validation scores, selected epoch, final test metrics, and four behavioral-pattern slices (not yet implemented here). Do not infer metric improvements without executing the run. Next milestone should add same-feature model comparisons and segmented evaluation before asserting a GRU advantage.
