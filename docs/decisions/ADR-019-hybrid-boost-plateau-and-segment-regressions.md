# ADR-019 — Record hybrid boost plateau and segment regressions

- Status: **Accepted as an evaluation interpretation; no model-deployment decision**
- Scope: v0.2.3 position + most-recent-hotel candidate-ranking baseline
- Related: ADR-018 (controlled hybrid design), Checkpoint A.2 taxonomy, v0.2.1 tie-neutral evaluation

## Context

A validation-tuned hybrid combines normalized recorded candidate position with a fixed boost for the last hotel referenced in retained history. The complete JSON evaluation yielded lambda=2 as the selected boost. We need to understand whether this is an identifiable optimal weight, and whether aggregate performance improvement generalizes across the established four behavioral segments.

## Evidence

- Candidate position scores span 0 to 1 (`1 - candidate_index/(candidate_count-1)` for multi-candidate lists).
- The evaluated lambda grid was `[0, 0.25, 0.5, 1, 2, 4]`; the selection rule maximizes **validation overall tie-neutral MRR**, then takes the smallest boost on ties.
- Validation MRR is 0.598401 for lambda=2 and lambda=4; their reported validation and test metrics are identical. Selected test MRR is 0.598007.
- Against position only, selected-hybrid **test** MRR changes by +0.480550 (immediate continuation), -0.137275 (return to earlier hotel), -0.044556 (new after exploration), and 0 (no prior hotel reference).
- Immediate-continuation test MRR is 1.000000 by design for an eligible label that matches the last hotel reference once its candidate receives a dominating boost.

## Decision

1. **Keep lambda=2** as the reproducible smallest selected boost on the observed plateau; do **not** call it a uniquely optimal intensity or a calibrated traveler preference.
2. **Retain tie-neutral MRR** as the primary offline baseline comparison. Candidate position is an explicit model feature in this experiment, rather than an implicit tie breaker.
3. **Require four-pattern reporting** alongside every overall score; no overall lift may be described without noting the regressions on return and new-hotel patterns.
4. **Classify the hybrid as candidate reranking**, not a pure traveler representation. Do not claim the target-row logged candidate order is verified available before the click-out in a serving environment.
5. **Do not start a new model in this ADR.** The next experiment requires its own hypothesis and predeclared validation-only selection criteria.

## Alternatives considered

- **Lambda=4:** Equivalent observed ranking behavior, less parsimonious number, rejected by tie rule.
- **Lambda=1:** Nearly matches overall MRR but is not identical; retains some different rankings and still incurs segment-level regressions.
- **Position alone:** Stronger on return and new-hotel patterns but lower on immediate continuation and aggregate MRR.
- **Adaptive gate / learned reranker:** Potentially helpful for pattern-specific trade-offs, but untested; must not use a future target-dependent pattern label at serving time.

## Consequences and review triggers

- Future comparisons should use the same chronological splits, candidate reconstruction, history eligibility, tie policy, and four-pattern segmentation.
- Before deployment-style claims, verify candidate-order availability at inference time and evaluate model behavior at explicitly defined prediction points.
- Reconsider this decision if a controlled, causally available context/behavior model improves held-out performance without unacceptable segment regressions.

See `docs/experiments/v0_2_3_position_behavior_hybrid_findings.md` for exact figures and interpretation.
