# ADR-015 — Characterize the prediction population before extending models

**Status:** Accepted for analysis protocol; substantive findings pending full-data execution.

## Context

The last-item baseline scores strongly on previously referenced targets but poorly on targets unseen in retained history. Existing seen/unseen numbers overlap with other history phenomena and do not explain first versus repeated click-outs, returns to earlier items or recent action types. Adding complex baselines before characterizing the benchmark would risk optimizing a poorly understood task.

## Decision

Pause model expansion. Audit every emitted click-out example using a four-way mutually exclusive hotel-reference taxonomy and a three-way prior-click-out taxonomy. Produce reproducible split-level counts, candidate-position and time-gap diagnostics, cross-tabulations, JSON and Markdown summaries. Match hotel-reference semantics to the *current* baseline code while recording that this choice remains provisional.

## Alternatives considered

- **Direct GRU training:** more model progress, but no answer to which behavior drives gains.
- **Sample session inspection only:** interpretable, but vulnerable to cherry-picking.
- **Analyze all raw events immediately:** broader denominator, but confuses raw-data eligibility with the existing benchmark. First audit the emitted population; then perform a separate raw-vs-processed coverage study.
- **Pattern classification using all user sessions:** possible longer-term target, but outside the current session-first contract.

## Boundaries and revision triggers

This is an example-weighted offline analysis, not a claim about independent users, booking or exposure. Treat target-row impression positions as descriptive, not causal. Revisit definitions if hotel-ID semantics, session ordering, or truncation bias is materially different from assumptions. No new model architecture is authorized by this ADR.
