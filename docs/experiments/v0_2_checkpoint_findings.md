# v0.2 / v0.2.1 — Baseline results and interpretation checkpoint

**Status:** Evidence recorded; no new modeling authorized by this checkpoint.
**Source:** Local terminal outputs and JSON excerpts shared during the TravelerRecLab project.
**Scope:** Trivago RecSys Challenge 2019 *next click-out among recorded candidates*. A click-out is not a completed booking.

## 1. Why we stopped here

We established a strong last-item baseline, then found that the original candidate-list order inflated metrics when models assigned tied scores. After tie-neutral evaluation, we discovered a dramatic gap between targets previously referenced in history and targets not previously referenced. This checkpoint fixes our interpretation before we add more features or a GRU.

## 2. Data and evaluation contract (observed vs assumed)

- Raw training events: **15,932,992**; users: **730,803**; user/session pairs: **910,683**.
- Click-out events: **1,586,586**; emitted history-eligible next-click-out examples: **1,276,216** (approximately 80.4% of click-outs).
- Most exclusions were short-history cases in the raw-order audit (**309,676**) plus **697** candidate-membership exclusions; source-order eligibility differed from sorted-builder output by **3**, which the later first-event reconciliation explained.
- Emitted example history lengths: median **5**, p90 **44**, with an imposed maximum of **50** events.
- In the full emitted corpus, the target was previously interacted with in about **46.67%** of examples. This is a **behavioral-repeat**, not a global item-cold-start, definition.
- Full validation: **229,804** examples; full test: **223,074** examples. Proposed chronological split: train November 1–4, validate November 5, test November 6, 2018. Confirm exact code configuration when reproducing.
- Each example ranks the clicked item among the **recorded click-out-row impression list**. The list is not independently proven to be available at an earlier serving timestamp; results are an offline proxy, not causal treatment effects.
- Ordering contract used by the builder: timestamp then step, with the implementation's deterministic tie behavior. Repeated `step=1` events across distant timestamps mean step-only sorting is unsafe; original-row-index tie-breaking remains a reproducibility recommendation to verify in code.
- Clicked hotel metadata coverage: **289,302 / 289,506 distinct clicked IDs (99.93%)**, with **99.96%** of click-outs on items with metadata. Historical metadata availability is unverified.

## 3. Full-model comparison

**MRR (higher is better).** Initial `v0.2` numbers used logged-list order to resolve tied predictions. The `v0.2.1` audit calculated tie-neutral expected reciprocal rank over tied positions. For models with ties, **tie-neutral MRR is the preferred primary comparison**.

| Baseline | Validation MRR (logged ties) | Test MRR (logged ties) | Test MRR (neutral ties) |
|---|---:|---:|---:|
| Random | 0.1731 | 0.1731 | Not reported in v0.2.1 excerpt |
| Position-only | 0.4206 | 0.4217 | **0.4217** |
| Global popularity | 0.2736 | 0.2696 | **0.2600** |
| Last referenced item | **0.5984** | **0.5980** | **0.5049** |
| Action-weighted history | 0.5579 | 0.5577 | **0.4771** |
| Recency-weighted, 5 minutes | 0.5776 | 0.5767 | **0.4944** |
| Recency-weighted, 1 hour | 0.5662 | 0.5658 | **0.4835** |
| Recency-weighted, 24 hours | 0.5652 | 0.5648 | **0.4824** |

The 5-minute recency setting was chosen on **validation tie-neutral MRR** from the tested grid `{300, 3600, 86400}` seconds. It is the best **tested** decay, not a proven global optimum. All these baselines were evaluated against the same candidate-ranking proxy.

### What this tells us

1. **Last-item is strongest overall among these baselines**, even after neutralizing ties: test MRR **0.5049**.
2. **Candidate position itself is predictive:** position-only MRR **0.4217**. This is inherited from the logged ranking, not a personalized embedding.
3. **Tie-breaking mattered materially:** last-item drops from **0.5980** to **0.5049**, a **0.0931** absolute difference when ties are treated neutrally. This is not a complete correction for exposure or selection bias.
4. **Short time decay helps:** 5-minute recency MRR **0.4944** exceeds 1-hour **0.4835**, suggesting strong short-term intent in this benchmark.
5. **The result does not prove neural sequence modeling is unnecessary.** We have not trained or evaluated a GRU or Transformer.

## 4. Segments reveal the key failure mode

The full test set partitions into targets previously referenced in the history and targets not previously referenced:

| Segment | Examples | Share | Position MRR | Last-item MRR | Action-weighted MRR | 5-min recency MRR |
|---|---:|---:|---:|---:|---:|---:|
| Target seen | 103,945 | 46.6% | 0.5129 | **0.9092** | 0.8588 | 0.8960 |
| Target unseen | 119,129 | 53.4% | **0.3421** | 0.1521 | 0.1440 | 0.1440 |
| All test examples | 223,074 | 100% | 0.4217 | **0.5049** | 0.4771 | 0.4944 |

All numbers in this table are **test tie-neutral MRR**. “Unseen” here means **not referenced in this example's available interaction history**, not a new hotel in the catalog or one missing training interactions.

**Interpretation:** The history-only models mostly assign meaningful scores to previously referenced hotels. When the clicked hotel is absent from the history, they cannot discriminate it well; many candidate scores tie. A contextual ranker or metadata-aware item representation might help, but that is an **untested hypothesis**, not a result.

### A metric trade-off on seen targets

On the *seen-target test segment*:

| Baseline | Tie-neutral MRR | Tie-neutral Recall@5 |
|---|---:|---:|
| Last-item | **0.9092** | 0.9168 |
| Action-weighted | 0.8588 | 0.9930 |
| Recency-weighted, 5 minutes | 0.8960 | **0.9954** |

Last-item excels at putting the correct hotel at the very top. Multi-item history aggregation retrieves more previously seen hotels in the top five. **Which baseline is best depends on the product task and metric.**

## 5. Why aggregate MRR was misleading

- The overall last-item MRR of ~0.505 combines extraordinarily high **seen-target** MRR (~0.909) with low **unseen-target** MRR (~0.152).
- The original 0.598 last-item MRR also partly inherited candidate order when the baseline couldn't distinguish candidates.
- Strong test performance on a history-eligible subset does not describe zero-history production traffic: approximately 19.6% of raw click-outs were not emitted, overwhelmingly for insufficient history under the audited predicates.
- Offline click-out ranking is not direct evidence of completed bookings or business-value lift.

## 6. What we do *not* know yet

- Whether a GRU or Transformer improves **unseen-in-history** relevance with *identical* candidate features and evaluation rules.
- Whether candidate metadata or search context, rather than sequence order, accounts for any future model lift.
- Whether historical metadata and logged candidate lists were present at a hypothetical production decision time.
- How much models improve business outcomes in a true online experiment.
- Confidence intervals for model differences; comparisons so far are point estimates.
- Full cold-start performance for zero-history events, since the main benchmark excludes them.
- Whether adding position features to simple baselines closes the gap to a more advanced ranker.

## 7. Frozen next-step criteria (not implementation)

Before comparing neural models:

1. Preserve **identical candidate lists and split boundaries** for all models.
2. Use **tie-neutral MRR** as the main offline diagnostic, and report Recall@5 and seen/unseen segments.
3. Tune on **validation only**; do not select architectures or hyperparameters using the test set.
4. Provide a **position/context baseline** when more context is added; otherwise a GRU may win from richer inputs rather than sequence order.
5. Keep an explicit **cold-start population** and acknowledge the source's inability to measure bookings.
6. Reuse the existing data contract and audit evidence rather than redefining labels silently.

**Checkpoint decision:** Document and understand these findings before making another modeling change.
