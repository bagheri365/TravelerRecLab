# Trivago EDA: evidence ledger before v0.2

**Status:** observations recorded; baseline experiments not yet run
**Source:** local `data/raw/train.csv`, `data/raw/item_metadata.csv`, and generated v0.1.5–v0.1.8 audits.
**Reproduce:** `python -m travelerreclab.data.inspect_trivago`; `python -m travelerreclab.data.eda_trivago`; `python -m travelerreclab.data.build_sequences`; `python -m travelerreclab.data.eda_sequences`; `python -m travelerreclab.data.validate_raw`; `python -m travelerreclab.data.audit_final`. Refer to the corresponding protocol docs for arguments.
**Policy:** numerical findings below are measurements; proposed model choices are hypotheses, not performance results.

## 1. Dataset scope and limitations

| Observation | Value | Interpretation |
|---|---:|---|
| Events | 15,932,992 | Ten action types in raw train CSV |
| Unique users | 730,803 | Anonymized IDs, not verified persistent people |
| Unique `(user_id, session_id)` keys | 910,683 | Session identifier should be scoped by user |
| Sessions with >=1 click-out | 826,842 | ~90.8% of observed session keys |
| Click-out events | 1,586,586 | Partner-offer click, **not a completed booking** |
| Event range (UTC) | Nov 1–6, 2018 | Six-day observation window; no long-term/seasonal validation |
| Median/p90/p99 events per session | 4 / 41 / 212 | Strong session-length skew |
| Median/p90/p99 sessions per user | 1 / 2 / 4 | Cross-session histories are sparse |
| Median/p90/p99 click-outs per click-out session | 1 / 4 / 10 | Multiple click-outs in some sessions |

**Available event columns:** `user_id`, `session_id`, `timestamp`, `step`, `action_type`, `reference`, `platform`, `city`, `device`, `current_filters`, `impressions`, `prices`.

`reference` is action-dependent (hotel, destination, filter, etc.); it must **not** be universally interpreted as a hotel ID. `impressions` and `prices` are candidate-list fields on click-out records and are not independent timestamped pre-click exposure observations. `item_metadata.csv` contains `item_id` and a `properties` string.

### Action mix

| Action | Count | Share |
|---|---:|---:|
| interaction item image | 11,860,750 | 74.44% |
| clickout item | 1,586,586 | 9.96% |
| filter selection | 695,917 | 4.37% |
| search for destination | 403,066 | 2.53% |
| change of sort order | 400,584 | 2.51% |
| interaction item info | 285,402 | 1.79% |
| interaction item rating | 217,246 | 1.36% |
| interaction item deals | 193,794 | 1.22% |
| search for item | 152,203 | 0.96% |
| search for poi | 137,444 | 0.86% |

Missingness is action-conditioned: `current_filters` is missing on ~92.76% of raw rows; `impressions` and `prices` on ~90.04%. All 1,586,586 click-outs have candidate lists, with no observed list-length mismatch or duplicate candidate items in the audit. Do not fill missing candidate fields from a later click-out.

## 2. Click-out label and candidate quality

| Measurement | Value | Denominator / caution |
|---|---:|---|
| Raw click-outs with candidates | 1,586,586 | All raw click-out rows |
| Raw target-in-candidates | 1,585,837 | 749 raw clicked IDs are absent from list |
| Candidate list median/p90/p99 | 25 / 25 / 25 | Candidate lists contain up to 25 hotels |
| Emitted next-click-out examples (`min_history=1`, `max_history=50`) | 1,276,216 | ~80.4% of raw click-out events |
| Source-order eligible examples | 1,276,213 | Not identical to generator's chronological sorting |
| Source-order excluded for insufficient history | 309,676 | Session-prefix rule, not necessarily first-time users |
| Source-order excluded for target missing from candidates | 697 | Exclusive predicate-order exclusion count |
| Difference (sorted minus source-order eligibility) | +3 | Explained by session first-event differences |

**Do not confuse these denominators:** 749 is a raw click-out membership check, while 697 is an exclusive *source-order pipeline exclusion reason* after other filtering. The generated-example candidate coverage is 100% by construction, not proof of exposure at prediction time.

**Training task proposed for v0.2:** Given only eligible pre-target history, rank the clicked hotel's ID among its **recorded** candidate set. This is a reconstructed offline click-out ranking benchmark, not an experimentally verified impression-time ranking dataset. The target click-out itself must not be in its own history. Define separate zero-history/context-only evaluation for the omitted population.

## 3. Sequence EDA and strong baseline hypotheses

The v0.1.6 diagnostic covers **emitted examples**, not every raw click-out:

| Observation | Reported result | Meaning |
|---|---:|---|
| Target previously referenced/interacted with | 595,567 (46.67%) | Prior interest could be a strong predictor; ensure action-aware ID typing |
| Last referenced item equals target | 521,428 (40.86%) | Motivates a last-item baseline; verify what is counted as an item reference |
| Target previously clicked out | 226,697 (17.76%) | Repeated click-outs are common |
| Truncated history-length p50/p90/p99 | 5 / 44 / 50 | p99 hits the imposed cap, **not** natural full-history p99 |
| Pre-Nov-5 / Nov-5-onward emitted examples | 823,338 / 452,878 | Diagnostic cutoff, not final three-way split |
| Users crossing diagnostic cutoff | 15,394 | Cross-period user activity is not itself leakage |
| Sessions crossing diagnostic cutoff | 6 | Check per-target temporal eligibility |

The repeat metrics overlap; do not add them together. The full event-history input includes many hotel-image interactions, so a learned sequence model must beat robust *non-sequential* baselines under the same candidates and split.

## 4. Ordering and causality

| Audit | Result |
|---|---:|
| Timestamp ties in source-order audit | 9,976,106 |
| Duplicate `(user, session, step)` rows | 657 |
| Timestamp reversals | 29 |
| Nonincreasing steps | 49 |
| Reentered session rows | 49 |
| Sessions whose first event changes under timestamp-based sorting | 27 |
| Sorted minus source-order eligible examples | +3 (reconciled) |
| v0.1.8 invalid identity/order rows | 0 |

The v0.1.8 JSON stores changed-session samples under **`ordering.sample_changed_sessions`**, not at the top level. Samples show repeated `step=1` with timestamps separated by hours or days. This means `step` cannot always be treated as a unique ordering key. For the *existing* benchmark, use the builder's deterministic **timestamp → step → original row order** contract, subject to code verification and sensitivity analysis. Treat the contract as a pragmatic reconstruction, **not** a proven event-availability sequence.

The three-count reconciliation is explained by first-event changes under sorting with `min_history=1`; it **does not** verify that all history contents and labels are perfectly ordered. Before model results are called leakage-safe, test monotonic target-relative event ordering and train/test cutoff behavior explicitly.

## 5. Hotel inventory, metadata and popularity

| Audit | Value |
|---|---:|
| Metadata rows/distinct hotels | 927,142 / 927,142 |
| Metadata duplicate IDs / missing property strings | 0 / 0 |
| Distinct clicked hotels | 289,506 |
| Distinct clicked hotels with metadata | 289,302 (99.9295%) |
| Click-outs on hotels with metadata | 1,585,943 / 1,586,586 (99.9595%) |
| Share of clicks on top 1% of clicked hotel IDs | 17.7503% |

Common metadata tags include `Satisfactory Rating`, `Car Park`, `Good Rating`, `WiFi (Rooms)`, `Shower`, `Television`, `WiFi (Public Areas)`, `Hotel`, `Very Good Rating`, and `Air Conditioning`. Tags are descriptive, and not necessarily historically valid at each decision time. Covering clicked IDs does not guarantee coverage of all candidate impression IDs. **Measure candidate-ID coverage separately** before metadata-dependent reranking claims.

## 6. Feature taxonomy for the initial benchmark

| Group | Raw sources | Initial encoding and precautions |
|---|---|---|
| Identity/grouping | user_id, session_id | Keys for grouping/splitting; avoid reliance on user-ID lookup for cold starts |
| Event identity | action_type, reference | Action embedding + typed reference; hotel IDs only for hotel-specific actions |
| Session context | platform, city, device | Categorical context; use values available at prefix time |
| Time | timestamp, step | Relative event gaps, recency, position; deterministic ordering policy |
| Search preferences | current_filters | Optional, action-specific; missing != no preference |
| Historical hotel activity | previous hotel-specific events | Last valid hotel, action counts, time-weighted scores, previous click-out flags |
| Candidate list | impressions | Offline reconstructed candidate ranking; exposure-time caveat |
| Candidate price | prices | Aligned by candidate position; no forward fill from future events |
| Property tags | item_metadata.properties | Later metadata-based hotel representations; measure candidate coverage and historical validity |

**Not available here:** verified completed bookings, saves, stay dates, real-time hotel inventory/availability, or long-term traveler histories beyond six observed days. Do not claim to evaluate booking propensity on this dataset.

## 7. Open decisions and v0.2 evaluation contract

1. **Ordering sensitivity:** compare `timestamp, step, source index` with defensible alternatives on anomalous sessions; assert prefix timestamp <= target timestamp and deterministic results.
2. **Split:** propose Nov 1–4 training, Nov 5 validation, Nov 6 test, based on **target time**. Fit global popularity and all tunable parameters using training data only. Earlier same-user history may be used at later prediction points only when causally available, consistent with stated serving scenario.
3. **Candidate contract:** reject/report invalid candidate target membership for eligible examples; keep identical candidate sets for all baselines; report target positional skew and explicitly flag observed-list availability limitation.
4. **Cold starts:** separate zero-history click-outs (~19.6% excluded overall for all causes) and unseen-item slices rather than hide them. Report coverage, not just conditional accuracy.
5. **Metric definitions:** MRR (primary), Recall@5/10 and NDCG@5/10 (supporting), deterministic tie breaks, per-segment metrics. One relevant click per ranked set makes MRR a meaningful starting point. Include confidence intervals by session where feasible.
6. **Baseline sequence:** uniform/rule-based sanity check, global training-popularity, last valid hotel, action-weighted hotel counts, exponential recency+action scoring; compare against GRU later. Test multiple decay constants **only on validation**.
7. **Coverage:** if a rule-based model cannot score a candidate (e.g., no prior hotel), define and report a deterministic fallback such as training-popularity, not an implicit zero-score tie.
8. **Reproducibility:** log dataset versions, filtering predicates, ordering rule, split thresholds, seeds, configuration, and selected hyperparameters.

### Interview summary (60 seconds)

> I audited about 15.9 million Trivago interactions and 1.59 million click-outs. Click-outs are hotel-offer selections, not bookings. Most sessions were short, the median user had one observed session, and roughly 47% of emitted targets referred to previously interacted-with hotels. That argues for a strong last-item and action-recency baseline before a GRU. I checked candidate lists and found near-complete raw clicked-item membership, but they are recorded on click-out rows, so I frame the task as an offline reconstructed ranking benchmark rather than verified impression-time exposure. I also found repeated steps and timestamp ties and documented a deterministic ordering policy with anomaly sensitivity tests. Hotel metadata covers almost all clicked items, enabling a later cold-start feature ablation. Finally, I use target-time chronological evaluation and a separate zero-history slice so improvements reflect real incremental value rather than leakage or selection bias.
