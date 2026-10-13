# Experiment template

## Identity
- **Experiment ID:**
- **Date / code commit:**
- **Author / machine / hardware:**
- **Status:** planned | running | completed | invalidated

## Research question and hypothesis
State a falsifiable hypothesis and why the business would care.

## Alternatives and controls
Specify baseline and treatment. Match data split, candidate set, feature availability, embedding size/parameter budget, training budget, and seeds where possible.

## Dataset and label semantics
- Data version and artifact checksum:
- Train / validation / test windows:
- Prediction timestamp and history cutoff:
- Positive label and negatives:
- Candidate provenance / position bias:
- Exclusions and population weights:

## Implementation and objective
- Input feature versions:
- Architecture and hyperparameters:
- Loss and sampling scheme:
- Training stop criterion:

## Metrics and uncertainty
- Primary metric (predeclared):
- Secondary metrics, slices, coverage:
- Statistical unit / confidence interval method:
- p50/p95/p99 inference and update latency:

## Results
| Model | Primary metric | Secondary metric | Runtime / cost | Notes |
|---|---:|---:|---:|---|
| Baseline | TBD | TBD | TBD | |
| Treatment | TBD | TBD | TBD | |

## Interpretation and decision
- What was learned, including negative results?
- Confounders / leakage checks / limitations:
- Decision: keep | revert | investigate
- ADR(s) updated:

## Interview explanation (60 seconds)
> Summarize the problem, baseline, measurement, trade-off, and next decision.
