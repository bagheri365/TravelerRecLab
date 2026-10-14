# ADR-010 — Reconcile event order and measure hotel metadata coverage

**Status:** Evidence collection / provisional until full v0.1.8 audit is reviewed.

## Context

The v0.1.7 audit measured 15,932,992 events, 1,586,586 click-outs, and 1,276,213 eligible examples in source order. The timestamp/step-sorted builder emitted 1,276,216. We also have 927,142 metadata rows but no confirmed clicked-hotel coverage.

## Decision

1. Explain the eligibility delta using a first-event comparison for `min_history=1`.
2. Do not silently change the sequence ordering convention before reviewing affected sessions.
3. Measure exact clicked-item metadata coverage, weighted and unweighted, plus property-tag frequencies.
4. Carry forward separate cold-start and history-eligible evaluation cohorts.

## Alternatives

- **Ignore the three-row discrepancy:** simpler, but reduces confidence in reproducibility.
- **Sort all 15.9M events again:** exact but more memory/compute intensive than the first-event identity for the current minimum-history rule.
- **Assume metadata coverage from catalog size:** incorrect; item-set intersection must be calculated.

## Trade-offs

The reconciliation is efficient but applies only to the current prefix eligibility conditions. The metadata check answers catalog coverage, not point-in-time feature availability. No approach here fixes the fact that click-out impression lists are not independently timestamped.

## Evidence and revisit criteria

Run `python -m travelerreclab.data.audit_final`, examine `sample_changed_sessions`, and confirm or investigate whether `predicted_sorted_minus_source_eligible == 3`. If not, inspect builder predicates and pandas parsing differences. Decide between timestamp-first and step-first using source documentation and anomalies, then regenerate sequences if the policy changes. Review metadata coverage before choosing ID-only versus metadata-backed hotel vectors.

## 60-second interview explanation

"I audited the event stream before training a sequential recommender. Some sessions contained ordering anomalies, and my raw-order eligibility count differed from the sorted training builder by three. Rather than dismissing the discrepancy, I implemented a reconciliation tied to the actual minimum-history policy. I also measured clicked-hotel metadata coverage at both the item and click-out levels, since a large metadata catalog doesn't guarantee coverage of the properties users actually select. This made my feature and evaluation choices auditable."
