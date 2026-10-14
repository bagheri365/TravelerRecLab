# Study Session 1 — Manually scoring candidate hotels

**Project:** TravelerRecLab
**Stage:** v0.2 / v0.2.1 learning checkpoint
**Type:** Educational walkthrough (NOT a new model experiment)
**Status:** Example calculations verified; final benchmark numbers transcribed from local experiment outputs shared in the project discussion.

## Learning objectives

After studying this example, explain: (1) an input prefix and label, (2) the scoring rule of each baseline, (3) MRR and neutral ties, (4) what scores do and do not mean, and (5) why unseen target hotels challenge history-only models.

## 1. Prediction point and data contract

**Constructed example, not an actual Trivago row.** A traveler explores hotels, then clicks out on Hotel B at time `t = 0`.

| Relative time | Action | `reference` |
|---|---|---|
| `t - 1200s` | `interaction item image` | Hotel A |
| `t - 900s` | `interaction item info` | Hotel A |
| `t - 30s` | `interaction item image` | Hotel B |
| `t` | `clickout item` | Hotel B (held-out label) |

The reconstructed target-row candidate list is `[C, A, B, D]`. **The target click-out must not enter its own history.** Candidate lists are logged with click-out rows; they are an offline benchmark assumption, not independently verified pre-click impressions.

We score *all four candidates* and compare the rank of B. Use hotel-typed references only (do not interpret destination searches or filters as hotel IDs).

## 2. Baseline scoring, by hand

| Baseline | Scoring rule | A | B | C | D |
|---|---|---:|---:|---:|---:|
| Position-only | `-candidate_position` | -2 | -3 | -1 | -4 |
| Global popularity | Training-period click-out counts (illustrative) | 20 | 10 | 50 | 5 |
| Last-item | `1` for most recent hotel-typed history reference; else `0` | 0 | 1 | 0 | 0 |
| Action-weighted | Sum event action weights per hotel | 2.5 | 1 | 0 | 0 |
| Recency-weighted (300s) | Sum `action_weight * exp(-age_seconds/300)` | 0.092996 | 0.904837 | 0 | 0 |

The illustrative popularity counts are **not from the dataset**. In the actual code, the popularity model is fit only on the training split. The action weights here match the project's initial heuristics: image = 1, info = 1.5. The 300-second decay was selected on validation among three trial values (300, 3,600, 86,400 seconds).

Arithmetic:

```text
A action total  = 1 + 1.5 = 2.5
B action total  = 1
A recency score = exp(-1200/300) + 1.5 * exp(-900/300)
                = 0.018316 + 0.074681 = 0.092996
B recency score = exp(-30/300) = 0.904837
```

## 3. Resulting ranks and per-example metrics

| Model | Ranking (ties indicated by braces) | Rank of B | RR = 1/rank |
|---|---|---:|---:|
| Position | C > A > B > D | 3 | 0.3333 |
| Popularity | C > A > B > D | 3 | 0.3333 |
| Last-item | B > {C, A, D} | 1 | 1.0000 |
| Action-weighted | A > B > {C, D} | 2 | 0.5000 |
| Recency-weighted | B > A > {C, D} | 1 | 1.0000 |

Because B is **not tied** under these example scores, logged-order vs tie-neutral reciprocal rank is identical for B. Over many examples, ties in the *target's* score matter significantly.

**MRR** is the mean of reciprocal ranks over many prediction examples; the numbers above are **single-example reciprocal ranks**, not dataset-level MRR. Recall@5 for each method is 1 in this four-candidate illustration (not useful for distinguishing them). For NDCG@K with one relevant target, the per-example value is `1/log2(rank+1)` if rank <= K; otherwise 0.

### Neutral-tie calculation example

Suppose all four candidate scores equal zero, with B at the recorded third position. Recorded-position tie-breaking gives `RR = 1/3`. Under a uniform random permutation of the four tied candidates, the **expected** RR is:

`(1 + 1/2 + 1/3 + 1/4) / 4 = 0.520833`.

This tie-neutral metric integrates over possible positions of the target within its tied block; it is **not** the same as assigning the target its mean rank and computing `1 / mean_rank`.

## 4. One key failure case: never-seen target

Keep the same history but switch the label to Hotel D, which does not appear in history. Last-item, action-weighted, and recency-weighted methods give D zero score; the candidate position and ties largely determine its eventual rank. These methods do not generalize from hotel A/B attributes to unseen hotel D. A candidate feature model or learned traveler-hotel similarity can potentially distinguish D from C, but must be evaluated fairly using equivalent available features across baselines.

**Important:** `target_unseen` in our segment report means **not previously interacted with by this session traveler** under the diagnostic's item-reference policy. It does **not** mean an entirely new hotel ID to the corpus.

## 5. Actual experiment results (not from the toy example)

On the *full test split* of **223,074** history-eligible candidate-ranking examples:

| Model | Tie-neutral MRR |
|---|---:|
| Position-only | 0.4217 |
| Popularity | 0.2600 |
| Last-item | 0.5049 |
| Action-weighted | 0.4771 |
| Recency-weighted, 300 seconds | 0.4944 |

For the last-item model, tie-neutral MRR was approximately **0.9092 on 103,945 previously seen targets** and **0.1521 on 119,129 unseen targets**. The difference is the primary reason to add a future experiment that handles unseen candidates using contextual and hotel metadata, before attributing any improvement solely to sequence order. The *position-only* model had about **0.3421** tie-neutral MRR on unseen targets.

**What these results establish:** The latest hotel reference is a strong predictor of repeat hotel click-outs, and logged candidate order materially affects evaluation when scores tie. **What they do not establish:** That GRU or Transformer is superior; that click-out means booking; that this offline candidate list was available at decision time; or that the results generalize to long-term traveler modeling.

## 6. Decisions and their status

| Decision | Reason | Status / revisit rule |
|---|---|---|
| Predict next click-out | Strongest observed Trivago selection event | Accepted for this dataset only; booking is not observed |
| Evaluate a fixed candidate set | Makes baseline ranking comparable | Accepted as offline reconstruction, not production replay |
| Restrict `reference` to hotel-typed actions | Avoid treating destinations as hotel IDs | Implemented in baseline scoring; keep explicit tests |
| Use last-item and action/recency pooling as controls | These test repetition and aggregation separately | Accepted; compare future neural methods fairly |
| Prefer tie-neutral MRR | Avoid benefiting from inherited logged order on equal scores | Accepted for comparative reporting; still has exposure bias |
| Use 300-second recency decay | Highest validation MRR of three settings tried | Provisional; do not call globally optimal |
| Analyze seen vs unseen separately | Aggregate MRR masks failure on unseen targets | Accepted; retain same cohort definitions |

## 7. Interview drill

**Q: Why can last-item beat a neural-looking aggregation model?**
A: It directly exploits the very recent property choice. Repeated interactions can make un-decayed aggregation favor an older hotel, while the actual click-out often targets the most recent hotel.

**Q: What exactly does MRR measure here?**
A: For each click-out event, rank recorded candidate hotels, take inverse rank of the clicked hotel, and average across examples. With score ties, use expected reciprocal rank under neutral ordering.

**Q: Why is unseen-target MRR weak?**
A: The history-only models primarily score hotel IDs already in the user's session. A previously unseen hotel gets zero personalized score, so context and hotel metadata are needed for generalization.

**Q: Is this proof GRU is necessary?**
A: No. First compare against strong candidate-context and metadata-aware *non-sequential* baselines, then isolate the effect of ordered history using equivalent inputs.

**Q: What is your leakage caveat?**
A: Prefix events are before the target under an explicit ordering rule, but candidate impressions are attached to the click-out row. They are acceptable for an explicitly labeled offline benchmark but not proof of a real-time pre-click candidate snapshot.

## 8. Self-check (try before looking above)

1. Why is the `clickout item` excluded from the event history when constructing this example?
2. Manually calculate A's recency score with a 300-second decay.
3. When does last-item assign a score of 1? Which event types can it use?
4. With equal scores for all four hotels, compare logged-position and tie-neutral RR for target B at position three.
5. Why might a model improve Recall@5 while worsening MRR?
6. Does unseen target mean the hotel is new to the entire dataset?

**Pause point:** Review and answer these questions before designing another model. Update any future decision record only after understanding the corresponding experiment.
