# v0.4 — Candidate-aware attention over GRU history states

**Status:** implementation available; real-data results pending.

## Motivation and hypothesis

In v0.3.1, the hotel-ID GRU reached test tie-neutral MRR 0.515058, versus
0.466170 for learned mean pooling. In the post-hoc order diagnostic, the same
GRU checkpoint reached 0.947021 on immediate continuation, but 0.290263 on
return-to-earlier-hotel and 0.171776 on new-hotel-after-exploration. Shuffling
histories hurt immediate continuation but improved earlier-hotel returns.

**Hypothesis:** letting each candidate attend to earlier GRU states can recover
previous interests without losing the final-state continuation signal. This is
an untested hypothesis, not a demonstrated improvement.

## Model

For each retained hotel-ID history, compute all GRU output states `h_1..h_T`.
For candidate `c`, form query `q_c = W_q e_c` and attention weights
`a_{c,t} = softmax_t(q_c^T h_t / sqrt(d))` over **only actual history tokens**.
Form `v_c = sum_t a_{c,t} h_t` and score

    s(c) = e_c^T (h_T + sigmoid(g) * v_c) / sqrt(d) + b_c

where `g` is a learned scalar initialized to 0. Empty hotel histories use the
existing learned `empty` vector and zero attention context. Shared candidate
embedding and bias, vocabulary, candidate softmax, tie-neutral metrics, and
split rules follow v0.3.1. The skip path helps retain recent-state information,
although **there is no guarantee** of unchanged continuation performance.

Attention is conditioned on candidate ID only, not on the true target label;
the clicked target enters only the cross-entropy supervision.

## Fairness and protocol

- Same eligible examples and Nov 1–4/Nov 5/Nov 6 chronological splits.
- Same train-only hotel-ID vocabulary and padding/UNK treatment.
- Same input hotel history as GRU v0.3.1 (no action type, price, metadata,
  destination, candidate position, or target-row context features).
- Same candidate-only cross-entropy loss, Adam, default dimension=32,
  learning rate=0.003, batch size=256, and three-epoch *initial* budget.
- Same validation-best checkpoint selection. Test is evaluated once by the
  training program after validation selection.
- More learnable parameters than GRU v0.3.1; not parameter-count matched.
- Do not use post-hoc test segment performance to tune architecture. The
  preceding segment audit motivated this preregistered hypothesis; validate
  before treating any segment change as confirmed.

## Run

```bash
python -m travelerreclab.models.gru_attention \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_attention_v04_smoke.json \
  --max-train 5000 --max-eval 1000 --epochs 2 --batch-size 128

python -m travelerreclab.models.gru_attention \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/gru_attention_v04.json \
  --epochs 3 --batch-size 256 --dim 32
```

The full run writes `gru_attention_v04.json` and `gru_attention_v04.pt`.
Monitor CPU/RAM because the vocabulary includes hundreds of thousands of IDs.
Do not compare the smoke MRR with the full-data GRU.

## Evaluation checklist

1. Check test population 223,074 and unknown-target/empty-history rates against GRU v0.3.1.
2. Compare validation trajectories and selected epoch with GRU 0.517260 validation MRR.
3. Compare full test tie-neutral MRR (GRU baseline 0.515058), Recall@5
   (0.603734), and NDCG@5 (0.510856); show absolute deltas.
4. Evaluate the same four behavioral segments using the selected checkpoint,
   without using test slices for iterative tuning. Existing GRU reference MRR:
   immediate 0.947021; earlier return 0.290263; new hotel 0.171776;
   no prior hotel reference 0.270072.
5. Record timing, peak memory, vocabulary size, model parameters, and random seed.
6. If comparing architecture definitively, repeat with matched validation-only
   training budgets and multiple seeds or confidence estimates clustered by session.

## Caveats

The candidate list is reconstructed from the target click-out row, not an
independently verified pre-click exposure. IDs unseen in training map to shared
UNK and cannot be distinguished by semantics. Multiple examples per session
are correlated. Three epochs need not establish convergence. No booking or
online conversion claims follow from offline click-out MRR.
