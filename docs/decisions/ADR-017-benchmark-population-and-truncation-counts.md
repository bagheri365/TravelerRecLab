# ADR-017 — Fix the benchmark population and distinguish raw versus retained histories

**Status:** Accepted for research reporting; no implementation or metric change.

## Context

Checkpoint A.3 reconciled 1,586,586 raw click-outs with 1,276,216 eligible, history-bearing prediction examples. It also resolved a seeming disagreement between 110,332 `truncated_if_emitted` and 110,223 `eligible_genuinely_truncated` events: **109** raw click-outs had >50 history events but failed later candidate eligibility checks.

## Decision

- Report all baseline metrics as **conditional on emitted examples** (at least one prior session event; click-out target represented in the candidate list with valid list alignment).
- Report first/repeat/switch proportions with an explicit **raw versus retained-history** population label.
- Never call an exactly-50-event history *truncated* unless its full length is independently known to exceed 50.
- Keep the provisional hotel-action interpretation, especially `search for item`, in evaluation caveats.
- Preserve the current dataset and baseline outputs unchanged. Do not redefine the benchmark retroactively to make raw-population percentages match emitted-population percentages.

## Alternatives and trade-offs

- **Expand the benchmark now to include zero-history click-outs:** rejected for the present comparison because historical baselines would require a new cold-start protocol and a new evaluation population. Keep it as a future, separate benchmark.
- **Treat all 50-event histories as truncated:** rejected because raw examples with exactly 50 prior events exist.
- **Treat metadata ID overlap as semantics proof:** rejected because overlap is evidence, not a guarantee.
- **Ignore the 109 difference:** rejected; accounting definitions matter for reproducibility.

## Consequences

Baseline results remain comparable within the fixed population. We explicitly recognize that the selected benchmark overrepresents subsequent click-outs relative to raw data. The A.2 retained-history `50_truncated` label should be interpreted as **retained length 50**, not actual truncation, until a separately scoped correction is implemented and tested.

Evidence and unresolved questions: `docs/study/checkpoint_a_03_raw_semantics_findings.md`.
