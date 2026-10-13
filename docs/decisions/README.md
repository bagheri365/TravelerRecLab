# Architecture Decision Record (ADR) index

This directory tracks *why* TravelerRecLab is built the way it is. Decisions are not the same as measured results.

## Status legend

- **Accepted:** foundational or constraint-driven choice; still subject to correction.
- **Provisional:** hypothesis/starting default requiring evidence.
- **Planned:** work is not yet implemented.
- **Superseded:** historical record; link to its replacement.

| ID | Subject | Status | How it can be revisited |
|---|---|---|---|
| [001](ADR-001-dataset-selection.md) | Trivago as primary dataset | Accepted | Data semantics or availability limitations |
| [002](ADR-002-prediction-target.md) | Next hotel click-out | Provisional | Compare against alternate targets |
| [003](ADR-003-causal-prefixes.md) | Session-local causal prefixes | Accepted | Test cross-session extension separately |
| [004](ADR-004-sequence-length.md) | 50-event truncation | Provisional | Compare 20/50/100 and truncation effects |
| [005](ADR-005-evaluation-and-baselines.md) | Temporal evaluation and simple baselines | Accepted | Audit ranking candidate provenance |
| [006](ADR-006-embedding-dimension.md) | Start at 128 dimensions | Provisional | Quality, storage, and latency sweep |
| [007](ADR-007-state-serving.md) | Incremental GRU state serving | Planned | Prototype parity and operational load |

## Review process

1. State the product decision and observable prediction timestamp.
2. List plausible alternatives, costs, and risks.
3. Choose an initial option and mark **Provisional** unless evidence already exists.
4. Name an experiment that could *falsify* the choice.
5. Record metrics and artifacts after the experiment; do not rewrite the original rationale.
6. Update status and date, or create a superseding ADR.

Use [the experiment template](../experiments/experiment_template.md) for each comparison.
