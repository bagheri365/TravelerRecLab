# ADR-011 — Freeze EDA evidence and baseline data contract

- **Status:** Provisional acceptance for v0.2; revisit after baseline diagnostics
- **Date:** 2026-10-13
- **Scope:** Trivago next-click-out offline ranking benchmark
- **Evidence:** [v0.1.8 EDA evidence ledger](../data/trivago_eda_findings_v0_1_8.md)

## Context

We have finished v0.1.5–v0.1.8 dataset auditing. We must avoid selecting architectures or interpreting metrics before fixing label definitions, population, feature availability, temporal ordering, and candidate semantics.

## Decisions

1. **Target:** next clicked-out hotel, not booking; predict from strictly preceding observed events.
2. **Population:** v0.1 generator emitted 1,276,216 examples with `min_history=1` and cap 50. Report this as *history-eligible*; separately measure zero-history and other exclusions.
3. **Candidates:** rank among the target-row recorded impression list for an *offline reconstructed* benchmark; do not state these are verified pre-click exposures.
4. **Ordering:** use the existing deterministic timestamp-first builder contract for reproducibility, pending anomaly-slice comparison. When implementing, explicitly check timestamp/step/source-index tie-breaking and actual pre-target feature filtering.
5. **Evaluation:** chronological target-time split; select hyperparameters on validation, report MRR, Recall@K, NDCG@K plus coverage and latency; no model fitting on test outcomes.
6. **Baselines first:** popularity, last valid hotel, action counts, recency/action weighting; future GRU/Transformer experiments must share features, examples, candidate lists, and evaluation code.
7. **Metadata:** property tags are viable for an ablation, not yet justified as time-valid online features; candidate coverage and item-cold-start evaluation remain open.

## Alternatives rejected for the first benchmark

- **Random train/test split:** overlapping temporal histories and over-optimistic estimation risk.
- **Treat click-out as booking:** unavailable labels, invalid business claim.
- **Step-only ordering:** repeated `step=1` observed in anomalous sessions.
- **Universal `reference` hotel embedding:** `reference` is action-dependent.
- **GRU/Transformer immediately:** would leave the sequential benefit unidentified relative to cheap recency rules.

## Consequences / risks

- The benchmark studies click-out selection, not downstream conversion.
- Short six-day coverage limits genuine long-term traveler generalization claims.
- Recorded candidate lists are not independently timestamped pre-click.
- The history-eligible population excludes many zero-prefix click-outs; performance must be reported with coverage.
- Reproducible sorting need not represent a perfect production event-availability order.

## Evidence required to graduate status to Accepted

- Verify actual builder tie-breaking and causal prefix invariants in tests.
- Fix chronological train/validation/test boundaries and audit examples crossing the split.
- Quantify valid scoring population and deterministic fallback behavior.
- Evaluate candidate-list position effects and item metadata coverage over **all candidate IDs**, not only clicked IDs.
- Document numeric baseline results and uncertainty before choosing sequential model complexity.

## Interview defense

> We froze a data contract after examining raw labels, short histories, repetitive hotel interest, sorting anomalies, and missingness. That makes subsequent baseline versus GRU comparisons fair. Our offline task is click-out ranking, not booking prediction, and we explicitly distinguish candidate lists recorded with a click-out from independently verified pre-click exposures. We use chronological evaluation with strict feature availability and evaluate cold-start coverage separately.
