# v0.3.1 — Controlled hotel-ID GRU session encoder

**Status:** implemented; full-data results not yet measured.

## Hypothesis

A GRU trained on hotel IDs in temporal order can learn order-sensitive traveler state and improve next-clickout ranking over v0.3 mean pooling. Whether it beats last-item, particularly on immediate continuation, is **unknown**.

## Protocol and architectural controls

- Same `next_clickout.jsonl` source, eligibility predicate, temporal split (train Nov 1–4; validation Nov 5; test Nov 6), 50-event retained-history cap, and train-only vocabulary as v0.3.
- Same hotel-specific event filtering (including provisional `search for item` semantics), padding/UNK handling, dim=32, shared hotel embedding and candidate-ID bias score `dot(z, e_h)/sqrt(dim)+b_h`.
- Same candidate-set cross-entropy objective and tie-neutral MRR, Recall@5 and NDCG@5. Best epoch selected on validation tie-neutral MRR, test run once after selection.
- **Only history encoder changes:** unweighted mean pooling to an `nn.GRU(dim, dim, batch_first=True)` whose final hidden state represents the traveler. Packed sequences suppress padding updates. Empty hotel histories use a learned vector.
- This is *GRU-style session encoding*, not a faithful reproduction of all GRU4Rec sampling, objective, and session-parallel training choices. No action embeddings, explicit time features, hotel metadata, or candidate position.
- As in v0.3, examples stream in file order; no randomized training order. Epoch budget is provisional and v0.3 validation MRR was still rising at epoch 3, so **three epochs cannot demonstrate a definitive architecture win/loss**.

## Run commands

Smoke test:

```bash
python -m travelerreclab.models.gru \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_v031_smoke.json \
  --max-train 5000 --max-eval 1000 --epochs 2 --batch-size 128
```

Full first-pass experiment:

```bash
python -m travelerreclab.models.gru \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_v031.json \
  --epochs 3 --batch-size 256 --dim 32
```

The report writes a `.pt` model checkpoint alongside its JSON metrics. PyTorch is an optional dependency (`python -m pip install torch`). Training on CPU may require substantial resources due to the hotel vocabulary and candidate softmax.

## Known reference results (not new findings)

- v0.3 hotel-ID mean pooling: validation MRR 0.467356; test MRR 0.466170, selected epoch 3 of 3, unknown-target rate 1.66%.
- Last-item heuristic: test tie-neutral MRR 0.504871.
- Position+last-item hybrid: test tie-neutral MRR 0.598007, **not feature-matched** because candidate position is explicitly included.

## Evaluation checklist

1. Verify that the train-vocabulary size and unknown/empty rates match v0.3.
2. Compare same-split validation learning curves; do not infer convergence from only three epochs.
3. Compare GRU to mean pooling, last-item, and candidate-position hybrid with their feature distinctions stated.
4. If the aggregate GRU result warrants investigation, add the **four predeclared behavior slices** in a later evaluation-only patch, without choosing hyperparameters using test-set slice results.
5. Record training time, device, checkpoint, and learning-rate/epoch limits. Repeat with matched tuning budgets if making an architecture claim.

## Leakage and deployment caveats

The Trivago candidate list comes from the target click-out record, without independent pre-click exposure timestamp. Historical target click-outs can legitimately occur in input history, but the current target click-out must not. Model scores do not include target click-out candidate position. This is an offline proxy task, not booking prediction or verified online personalization lift.
