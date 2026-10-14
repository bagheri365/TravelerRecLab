# ADR-009 — Validate raw sequences and exclusions before training baselines

**Status:** Accepted for process; modeling consequences pending full-data results.

## Context

Our source contains 1,586,586 click-outs but v0.1 emits 1,276,216 examples. The difference is 310,370. The previous EDA on emitted examples cannot identify missing-example reasons. Sequence examples may also be sensitive to raw order and timestamp/step sorting.

## Decision

Run three targeted source audits before finalizing v0.2: source-order anomalies, mutually exclusive click-out exclusion reasons, and hotel metadata/popularity properties. Record actual findings before assigning modeling consequences.

## Alternatives

- Skip validation and proceed to GRU: faster but can hide selection and label-quality problems.
- Audit only emitted examples: easy, but cannot explain omitted targets.
- Immediately redesign the pipeline: premature without observed failure modes.

## Trade-offs

The audit takes additional runtime and session-level memory, and source-order exclusion counts can diverge from the sorted builder. In return, it makes the data contract and coverage measurable.

## Revisit criteria

Update this ADR after inspecting `data/processed/raw_validation/report.md`. Resolve any discrepancies between raw-source eligibility and emitted output before claiming complete exclusion accounting. Establish a cold-start and metadata coverage plan.

## Interview-ready explanation

> Before benchmarking sequential architectures, I audited raw event ordering and training-example exclusions. This matters because an offline improvement on a history-rich subset might not generalize to cold-start traveler sessions. The audit separates verified data-quality findings from assumptions about serving-time feature availability.
