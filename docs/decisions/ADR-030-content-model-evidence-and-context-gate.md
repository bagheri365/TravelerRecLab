# ADR-030 — Interpret property-aware attention-GRU as a small improvement, prioritize context verification

Status: **Accepted research interpretation; next architecture not yet authorized**
Date: 2026-10-14
Related: ADR-028 (metadata feasibility), ADR-029 (content-aware model), v0.5.1 full findings and diagnostics.

## Context

The v0.5 metadata audit established near-complete property coverage but also that most hotels new to a traveler's retained history are *not* unknown to the training vocabulary. The v0.5.1 ID-plus-property attention-GRU was intended to improve selection of such new-to-history hotels, while preserving continuation and earlier-return performance.

## Evidence

- Test overall MRR: v0.4 **0.520126**, v0.5.1 **0.521199** (Δ **+0.001074**); validation Δ **+0.000949**.
- New-hotel-after-exploration test MRR: **0.172012 → 0.173175** (Δ **+0.001163**); validation **0.173670 → 0.175063**.
- Return-to-earlier-hotel test MRR: **0.359127 → 0.358356** (Δ **−0.000771**); continuation test MRR **0.949408 → 0.950185**.
- v0.5.1 test new-to-history target MRR, by training exposure: clicked in train **0.183486** (71,467), in-vocabulary but never clicked **0.121731** (13,710), outside train vocabulary **0.153476** (1,606). These are different populations, not matched counterfactuals.
- Direct v0.4 per-exposure scoring on the same examples is unavailable. Consequently, a cold-start gain over v0.4 **cannot** be claimed.

## Decision

1. Record v0.5.1 as a **small, repeatable-across-splits descriptive improvement**, not a large discovery-model breakthrough or established production lift.
2. Classify the new-to-history hypothesis as **weakly supported** and the true unseen-ID/cold-start improvement as **not established**.
3. Preserve the behavioral regressions and exposure-group findings in reporting; do not optimize new hyperparameters against test diagnostics.
4. **Before adding another model**, audit contextual-feature semantics, prediction-time availability, and feature leakage, especially destination and filters. Candidate position and historical metadata availability remain unverified for production.
5. Keep attention-GRU v0.4 as the feature-controlled ID-only reference and retain the v0.5.1 content-aware result as a separate extension.

## Alternatives not selected

- Declare property embeddings a solution to cold start based only on out-of-vocabulary slice MRR: **rejected** because no matched v0.4 comparison exists.
- Immediately train a larger Transformer: **deferred**; current failure concerns candidate/context discrimination, not proof that sequence modeling capacity is insufficient.
- Conclude that properties have no value: **rejected**; the measured effect is small but positive across both chronological aggregate splits.

## Consequences and validity limits

The next research gate is a leakage-aware **search-context availability audit**, not automatically a new architecture. The data remains a target-row candidate-ranking reconstruction. v0.4 and v0.5.1 differ in parameters and scoring pathways, and there are no repeated-seed or session-clustered confidence estimates. Metadata was not point-in-time validated. Final model selection must remain validation-driven with the test set reserved for a locked comparison.

See `docs/experiments/v0_5_1_content_behavioral_and_exposure_findings.md` for full MRR/Recall@5/NDCG@5 tables, exact counts, and reproduction commands.
