# ADR-025 — Candidate-aware attention over historical GRU states

**Status:** Proposed for controlled v0.4 experiment.

## Context

The v0.3.1 GRU improves aggregate history-only ranking but concentrates much
of its performance on immediate hotel continuation. The order diagnostic
reveals poorer return-to-earlier-hotel performance, suggesting that relying
on a single final history state may be restrictive. The diagnostic is
post-hoc evidence about ranking behavior, not proof of internal causality.

## Decision

Keep the input, train-only hotel vocabulary, objective, metrics, and temporal
splits unchanged. Add candidate-conditioned dot-product attention over **all**
valid GRU history states, with a learned-scaled final-state skip. Compare
validation-selected v0.4 to the saved v0.3.1 reference. Report both aggregate
and behavioral results, including regressions, with no test-driven retuning.

## Alternatives

- Replace the GRU with a Transformer: changes multiple architecture choices
  at once, so defer until attention-on-GRU is evaluated.
- Add candidate position: potentially useful, but confounds assessment of
  learned traveler state, and pre-click availability is unverified.
- Add hotel metadata/context: needed for cold-start hotels, but separate from
  the question of retrieving earlier *known* hotel interests.
- Use only attention without last-state skip: may sacrifice high continuation
  performance; a skip is a deliberate modeling assumption, not a guarantee.

## Consequences and exit criteria

Attention adds parameters and runtime proportional to history length times
candidate count. It cannot fix hotel-ID cold-start limitations. Success means
reproducible, leakage-controlled training and **measured** validation and test
comparisons, not necessarily outperforming v0.3.1. Before claiming that
attention is beneficial, inspect the complete validation/test metrics and
behavioral trade-offs, with equal-feature and budget caveats.
