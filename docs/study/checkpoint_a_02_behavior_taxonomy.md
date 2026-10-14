# Checkpoint A.2 — Systematic behavioral-pattern taxonomy

**Status:** Analysis protocol implemented; full dataset findings pending local run. **This is not a new model.**

## Research question

What share of our *emitted next-click-out examples* represent repeated hotel interest, a return to an earlier property, a new hotel after exploration, or a click-out with no prior hotel reference? How do these patterns vary over time and by preceding click-out behavior, action, history length, recency, and logged candidate position?

## Hypotheses (not findings)

1. Immediate continuation is common and partly explains the last-item baseline's high MRR.
2. Return-to-earlier hotel may favor multi-event aggregation over last-item.
3. New hotel after exploration may favor candidate-position/context baselines.
4. Patterns may vary between Nov 1–4, Nov 5, and Nov 6.

## Population and mutually exclusive rules

One example is an eligible `clickout item` record in `next_clickout.jsonl`, with an available prior history (1–50 retained events), valid candidate array and clicked target. **This is not a distribution over all raw events, travelers, sessions, or click-outs**; excluded zero-history/malformed-candidate decisions are missing.

Treat as hotel references only `interaction item image/info/rating/deals`, `clickout item`, and provisionally `search for item`, consistent with `evaluation.baselines.HOTEL_ACTIONS`. Destination/POI/filter references are excluded from hotel identity comparisons. The `search for item` semantics remain unverified and require sensitivity analysis.

The four *exclusive* categories (check in this order):

1. **Immediate continuation:** target equals the **most recent hotel-referencing event**, not necessarily the latest event overall.
2. **Return to earlier hotel:** target was in retained hotel references but differs from the most recent one.
3. **New hotel after exploration:** some hotel references exist, but none match the target.
4. **No prior hotel reference:** retained history has no hotel-typed reference.

An orthogonal, exclusive prior-click-out classification:

- **First click-out:** no earlier `clickout item` in retained history.
- **Repeat click-out:** last earlier click-out targets the same hotel.
- **Switch click-out:** last earlier click-out targets a different hotel.

Be precise: **first** means first *in retained history*, not necessarily the true first in a potentially truncated session. Likewise, **immediate continuation** is a hotel-relative definition: intervening destination searches or filters do not break it.

## Summary and diagnostic statistics

Report each category's count, share and total, separately for all emitted examples and UTC train/validation/test. Also record number of distinct `(user_id, session_id)` combinations as *context*, not an independent-session pattern distribution. Analyze:

- Preceding action (`history[-1].action_type`), previous-click-out category.
- Retained history length: 1–5, 6–20, 21–49, 50 (potentially truncated).
- Distinct referenced hotels: 0, 1, 2–5, 6+.
- Target's logged candidate position: 1, 2–5, 6–10, 11+.
- Time since last event, last hotel reference, last target reference (or `not_seen`).
- Cross-tabs: hotel behavior × prior click-out / history length / last action / position / last-event gap.

Time-gap buckets: 0s, 1–10s, 11–60s, 61–300s, 301–3600s, >3600s. Because timestamp ties exist, 0s does not prove an action preceded another in actual human time.

**Denominators:** percent is always count divided by number of emitted examples *within that split*, never all raw click-outs. Cross-tabs in JSON are raw counts, not conditional percentages; calculate row shares from the JSON if needed. Multi-click-out sessions produce correlated observations.

## Reproduce

```bash
python -m travelerreclab.data.audit_behavior_patterns \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/behavior_patterns_a2_smoke.json \
  --report data/processed/behavior_patterns_a2_smoke.md \
  --max-examples 10000

python -m travelerreclab.data.audit_behavior_patterns \
  --input data/processed/next_clickout.jsonl \
  --output data/processed/behavior_patterns_a2.json \
  --report data/processed/behavior_patterns_a2.md
```

The smoke-test cap is the first N file records **in total**, not an unbiased or per-split sample. Only full-data output should be used for conclusions; a smoke run can omit validation/test entirely. Outputs contain no raw user or hotel IDs. This script reads JSONL without modifying source examples or models.

## Integrity checks / acceptance criteria

- Counts of four patterns sum to `n_examples` within every split.
- Counts of prior-click-out categories also sum to `n_examples`.
- All cross-tab cells are compatible with their pattern totals.
- Full run should enumerate the same 1,276,216 emitted examples if using the original processed artifact. Differences require investigating data versions.
- History timestamps must not be after target timestamps; malformed examples fail clearly.
- Compare split distributions before interpreting stable evaluation metrics.
- **Do not publish any measured pattern percentages until full local output exists.**

## Limits, alternatives and follow-ups

- **Truncation:** retained 50-event history can hide earlier hotel or click-out events, so `new` is *new in retained history* only. A separate raw-session audit is required to measure disagreement.
- **Ordering:** inherits upstream `(timestamp, step, source row index)` policy; tie anomalies are known. Sensitivity to ordering remains open.
- **Causality:** only previous events classify patterns; candidate position is a diagnostic of the *target click-out row*, **not** an independently verified pre-decision feature.
- **Interpretation:** an earlier click-out means selection of an outbound link, not a booking.
- **Coverage:** benchmark does not include zero-history click-outs; require a distinct raw population audit before discussing all travelers.
- **Prior-click-out segmentation** may be wrong for long sessions with earlier click-outs truncated away.
- **Hotel identity semantics:** a `search for item` reference has not been externally validated; repeat classification excluding this action if needed.

## Decision status

**Accepted:** measure these four mutually exclusive patterns and orthogonal click-out categories across chronological splits. **Provisional:** hotel-typed action taxonomy and retained-history definition of first/new. **Not decided:** any additional baseline model or architecture choice. Do not use this audit alone to claim preference inference or model lift.

## Interview-ready framing

> I first determined the composition of the offline prediction population before expanding my baseline suite. I separated immediate repeat-interest cases, returns to earlier hotels, truly new-to-retained-history targets, and examples without prior hotel signals. I independently measured first/repeat/switch click-outs and checked how these patterns differed by action type, candidate position, recency, and time split. I explicitly noted that truncated histories and target-row candidate lists limit what the findings mean for production.
