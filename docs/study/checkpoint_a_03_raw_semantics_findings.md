# Checkpoint A.3 — Raw semantics and benchmark-population findings

**Status:** Full raw audit executed; interpretation documented; no modeling change.

## Question and source

How do the raw click-out population, hotel-reference semantics, and full session histories differ from the emitted next-click-out examples? Source: full `data/raw/train.csv` audit via `python -m travelerreclab.data.audit_raw_semantics --input data/raw/train.csv --metadata data/raw/item_metadata.csv --output data/processed/raw_semantics_a3.json`, and the previously generated A.2 behavioral audit. Values below are observed output supplied from the full-data run; the raw report itself is not committed (large generated data remain local).

The audit uses `(user_id, session_id, timestamp, step, source row)` ordering. Historical hotel actions provisionally include `search for item`. Trivago has click-out labels, **not bookings**, and candidate lists are attached to target click-out rows, rather than independently verified pre-click impressions.

## 1. Inclusion contract and reconciliation

The unit of the **raw audit** is a click-out event. The unit of the **baseline benchmark** is an emitted click-out example with at least one earlier session event and a target hotel appearing in a candidate list with aligned prices. Earlier events are computed under the stated ordering, and the emitted history retains at most 50 events. The 697 non-emitted history-bearing examples are consistent with the previously audited candidate-target exclusion count; the eligibility code also checks a nonempty reference, nonempty candidates and aligned candidate/price lengths, so the final reason breakdown should be confirmed against that builder rather than inferred solely from this aggregate.

| Population | Count | Relationship |
|---|---:|---|
| All raw click-outs | 1,586,586 | Starting population |
| Zero preceding events | 309,673 | Excluded by `min_history=1` |
| With preceding events | 1,276,913 | Raw click-outs eligible for history requirement |
| Emitted benchmark examples | 1,276,216 | Pass history and candidate eligibility |
| With history but not emitted | 697 | 1,276,913 − 1,276,216 |

Checks: 309,673 + 1,276,913 = 1,586,586; 1,276,216 + 697 = 1,276,913. Both the script's click-out-type and history partitions reported `True`.

**Scope constraint:** Benchmark metrics do not characterize zero-history click-outs. The processed population is selected and should not be described as representative of every raw click-out.

## 2. Prior click-out semantics

Raw session histories (untruncated) give:

| Full-raw category | Count | Share of 1,586,586 |
|---|---:|---:|
| First click-out in the `(user_id, session_id)` pair | 826,842 | 52.11% |
| Same hotel as most recent prior click-out | 186,861 | 11.78% |
| Different hotel from most recent prior click-out | 572,883 | 36.11% |

In the A.2 emitted-example analysis of retained histories: first 41.79%, repeat 14.53%, switch 43.68%. **Do not interpret this difference as behavioral drift:** changing the inclusion population and available history changes the composition. A switch of clicked hotel does not imply rejection or purchase.

## 3. `search for item` reference semantics

The raw audit measured 152,203 `search for item` actions, all with nonempty references; 151,894 references matched `item_metadata.item_id` and 309 did not. Membership rate = **99.797% (approximately 99.80%)**.

**Evidence:** This is strong support for treating the field as a hotel ID for this action. **Limitation:** Metadata membership does not prove all event semantics; non-matches could include missing metadata or other values. Keep hotel-action classification **provisional** and retain sensitivity analysis as an open option.

## 4. Truncation counters — explain the 109 difference

The audit implementation increments `truncated_if_emitted` when a raw click-out has more than 50 preceding events, **before checking target/candidate eligibility**. Only afterward, if the example passes eligibility, it increments `eligible_genuinely_truncated`.

| Raw audit counter | Count | Exact operational meaning |
|---|---:|---|
| `below_max_history_before_target` | 1,164,154 | Click-outs with history lengths 1–49; prior to candidate eligibility |
| `exactly_max_history_before_target` | 2,427 | Click-outs with exactly 50 preceding events; not actually truncated |
| `truncated_if_emitted` | 110,332 | Click-outs with >50 prior events, **regardless of candidate eligibility** |
| `eligible_genuinely_truncated` | 110,223 | >50 prior events **and** valid emitted-example eligibility |
| Difference | **109** | >50-history click-outs filtered by eligibility |

Reconciliation: 1,164,154 + 2,427 + 110,332 = 1,276,913 history-bearing raw click-outs. For the >50 group, 110,332 − 110,223 = 109.

**Naming issue:** `truncated_if_emitted` sounds like an emitted-example counter but includes ineligible click-outs. Read it as *would require truncation if emitted*. Do not compare it directly to the A.2 `50_truncated` bucket, whose definition is retained length **equal to** 50, not proof of actual truncation. Any difference in exact-50 counts is affected by eligibility and warrants checking the joint raw eligibility/length table before concluding otherwise.

## 5. Target identity lost by truncation

- `eligible_target_seen_full_history`: **599,616** eligible examples whose target occurs in a hotel-typed reference anywhere before the click-out.
- `eligible_target_lost_by_truncation`: **4,049** eligible examples whose target appears in full history but not in the last 50 events.
- The A.2 retained-history `not_seen` count is **680,649**. Thus **4,049 / 680,649 ≈ 0.595%** of retained-unseen labels are older known hotels omitted from retained history.

The 4,049 correspond to approximately **0.317%** of all eligible examples. Hence truncation changes some seen/unseen labels but cannot, on its own, account for most retained-unseen targets. Raw/processed definitions should still be compared with the same provisional hotel-action taxonomy.

## 6. Research implications (hypotheses, not measured causal claims)

| Measured evidence | Interpretation | Limitation / alternative | Open question |
|---|---|---|---|
| 309,673 click-outs have zero history | A history encoder cannot use a prior session event for these points | Search context or candidate features may still exist | How should an online system serve a zero-history traveler? |
| Switch share rises from 36.11% raw to 43.68% emitted | Selection changes the mix of click-out behaviors | Raw and retained-history categories have different scopes | Does baseline performance vary by full-history click-out class? |
| 99.80% `search for item` ID membership | Hotel action assumption is supported | Membership is not intent validation | Would excluding nonmatching references alter A.2 patterns? |
| 110,223 eligible examples genuinely exceed 50 events | Capping can discard evidence | Exactly 50 events is not truncation | How sensitive are models to 20/50/100 event caps? |
| 4,049 labels move from seen in full history to unseen in retained | Truncation affects a small subset of our repeat/new taxonomy | Other hotel actions might be misclassified | Does correcting this change segment-level conclusions? |

## 7. Current decisions and stopping point

1. Freeze baseline evaluation **population** as the existing, explicit history-eligible candidate-set benchmark for reproducibility; do not silently broaden its interpretation.
2. Document raw and emitted populations separately in future results.
3. Treat `search for item` as a provisional hotel reference strongly supported by the membership audit.
4. Use `history_length_50` for retained-length-only buckets, and reserve *genuinely truncated* for examples with >50 preceding events confirmed in raw data.
5. No additional model complexity is justified by this audit alone. Complete the data/problem understanding and then review baseline coverage.

**Not established:** confirmed impression exposure timing, booking outcomes, production online feature availability, causal effects of candidate position, or generalization beyond this six-day logging period.
