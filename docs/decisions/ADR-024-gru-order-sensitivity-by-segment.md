# ADR-024 — Interpret GRU order sensitivity by behavioral segment

Status: **Accepted as diagnostic interpretation; no new model approved by this ADR**.

## Context

The v0.3.1 GRU reached 0.515058 tie-neutral test MRR (223,074 examples). The same fixed checkpoint evaluated with one seeded order permutation scored 0.449892. The accompanying behavioral analysis is recorded in `docs/experiments/v0_3_1_gru_behavioral_findings.md` and the generated `gru_v031_{validation,test}_diagnostics.json` reports.

## Decision

1. Interpret evaluation-time shuffling as evidence of **order sensitivity of this trained checkpoint**, not as proof of a causal advantage from sequential modeling over other architectures.
2. Report results for four behavioral segments alongside aggregate metrics. Immediate continuation gains cannot stand in for robust traveler-interest modeling.
3. Describe a **possible** overemphasis on recent hotel identity, rather than asserting that particular gate behavior has been demonstrated.
4. Keep the v0.3.1 GRU as the current history-only learned sequential reference; do **not** declare the position-augmented hybrid directly comparable on inputs.
5. Do not tune on the already inspected test-set breakdown. Choose future architectures and hyperparameters using validation, then run test once per locked comparison.

## Consequences

- The next modeling hypothesis should target earlier-hotel returns and new-to-history candidates **without sacrificing** immediate continuations.
- An attention-over-history encoder may help retrieve older hotel signals, but hotel-ID-only attention will not solve cold-start candidate feature gaps.
- A shuffled-history-trained control, repeated shuffle seeds, subgroup diagnostics, and session-clustered confidence intervals remain future work, not prerequisites falsely marked as completed.
- Preserve reproducibility with the checkpoint, explicit seed, and reports; do not treat the diagnostic as an online impact estimate.
