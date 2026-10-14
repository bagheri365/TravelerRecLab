# ADR-028: Treat hotel metadata as feasible candidate content, not proof of cold-start improvement

- **Status:** Accepted for the *next controlled experiment* (not a decision to deploy).
- **Context:** v0.5 metadata coverage audit, following v0.4 attention-GRU behavioral diagnostics.
- **Evidence:** `data/processed/hotel_metadata_audit_v05.json`; detailed findings in `docs/experiments/v0_5_metadata_coverage_findings.md`.

## Context

The v0.4 attention-GRU reaches test tie-neutral MRR **0.5201**, but only **0.1720** on `new_hotel_after_exploration`. We suspected unavailable ID embeddings could be a cause. The audit instead finds **85,177/86,783 (98.15%)** of these targets are already in the training vocabulary; **71,467/86,783 (82.35%)** appeared as positive training targets. True ID-vocabulary-unknown targets number **1,606** in this segment.

Metadata provides nonempty properties for **86,735/86,783 (99.94%)** of these targets. Across all test targets, **222,984/223,074 (99.96%)** have metadata; test candidate-slot coverage is **5,140,147/5,142,220 (99.96%)**. The relevant property vocabulary has **157 tags**, with a median of **17** tags per relevant hotel.

## Decision

1. Close the **metadata coverage audit** as a descriptive milestone, with full denominators and constraints recorded.
2. Author a future **ID-plus-property candidate-encoder** experiment, preserving the existing attention-GRU history encoder, training-only hotel-ID vocabulary construction, candidate-softmax objective, chronology, and evaluation protocol as far as practical. Clearly document any changed feature path, additional parameters, or training procedure.
3. Evaluate on **new-to-retained-history targets** separately from **absent-from-train-vocabulary targets** and **never-positive-in-training targets**. Report validation and test segment metrics; use validation alone for model selection.
4. Treat historical availability of snapshot metadata as **unverified**; do not make production or leakage-free claims based solely on this audit. Implement missing-feature fallbacks before model comparison.
5. Do **not** conclude that property embeddings will fix the weak new-hotel ranking or that all novel-to-session hotels are ID cold-start.

## Alternatives considered

- **Transformer first:** deferred; changing sequence capacity does not directly answer whether hotel content improves ranking of previously unconsidered candidates.
- **Property-only model first:** useful later as an ablation, but not a minimal extension of the current best ID-based neural reference.
- **Adding logged candidate position:** potentially strong reranking baseline, but different input contract and unverified pre-click availability. Keep separate.
- **Assume missing embeddings explain new-hotel failures:** rejected as primary explanation by the measured 98.15% train-vocabulary membership.

## Acceptance criteria for follow-on experiment

Document feature vocabulary and join semantics; preserve test holdout for reporting rather than hyperparameter search; compare to **v0.4 ID-only**; report the four behavior groups and target train-vocabulary / train-positive strata; include cold-start and missing-metadata fallback tests; record training duration, parameter differences, and uncertainty about historical feature availability. Do not represent aggregate improvement alone as proof of content generalization.

## Consequences

- We have a concrete, data-supported reason to test property features without confusing *new to traveler* and *unknown to training*.
- Success remains an open empirical question; content availability is not evidence of content usefulness.
- A proper production design will still require timestamp-consistent feature snapshots, context features, and independent candidate exposure information.
