# v0.2.3 — Position + last-hotel hybrid: findings

Status: **experiment completed; interpretation recorded; no new model authorized**.

## Research question and scope

Can a fixed boost for the most recently referenced hotel improve the ranking of the candidate hotels recorded on the target click-out row? This is a **candidate reranking** diagnostic, not a demonstration that a general-purpose traveler embedding has been learned.

Input: `data/processed/next_clickout.jsonl` generated from Trivago RecSys Challenge 2019 training events. Evaluation uses the established chronological validation (2018-11-05 UTC) and test (2018-11-06 UTC onward), eligible examples with retained 1–50 preceding events, and target present in the reconstructed candidate set. Validation has 229,804 examples; test has 223,074. There may be multiple correlated examples per session.

## Scoring rule

For candidate `h_j` with zero-based candidate index `j` among `K` candidates:

- Position score: `p_j = 1 - j / (K - 1)` for `K > 1`; implementation behavior for `K=1` should be preserved as tested.
- Behavioral score: indicator that the candidate equals the most recent hotel-specific reference in the **retained** history.
- Hybrid: `s(h_j) = p_j + lambda * 1[h_j == h_last]`.

For `K>1`, `p_j` lies between 0 and 1. For boosts strictly greater than 1, a candidate matching `h_last` necessarily outranks every nonmatching candidate, if that candidate is in the candidate list and hotel-identity handling matches this rule. Thus lambda=2 and lambda=4 induce identical rankings. Neither number is a uniquely estimated preference weight. Check exact code semantics for `K=1` and missing recent hotel reference when making implementation-level guarantees.

Metric: **tie-neutral MRR** is the primary comparison; for tied model scores, reciprocal rank is averaged over tied positions rather than silently inheriting the original candidate order. Recall@5 and NDCG@5 are also available in the JSON. Hyperparameters were selected using overall validation tie-neutral MRR, with the smallest boost breaking validation ties. Do not select a setting using test metrics.

## Full-data boost sweep

| Boost | Validation MRR | Test MRR |
|---:|---:|---:|
| 0 | 0.420565 | 0.421720 |
| 0.25 | 0.501567 | 0.501537 |
| 0.5 | 0.548440 | 0.547104 |
| 1 | 0.597033 | 0.596477 |
| **2 (selected)** | **0.598401** | **0.598007** |
| 4 | 0.598401 | 0.598007 |

Lambda 2 and 4 have identical reported aggregate and segment-level evaluation metrics, including Recall@5 and NDCG@5, on both splits. Lambda 2 was selected by the documented tie rule. Lambda 1 nearly saturates aggregate performance (0.001367 lower validation MRR) and is not equivalent on all examples.

## Segment-level results: full test set

| Pattern | n | Position-only MRR (lambda=0) | Selected hybrid MRR (lambda=2) | Delta |
|---|---:|---:|---:|---:|
| Immediate continuation | 93,005 | 0.519450 | 1.000000 | +0.480550 |
| Return to earlier hotel | 10,940 | 0.457387 | 0.320111 | -0.137275 |
| New hotel after exploration | 86,783 | 0.299414 | 0.254858 | -0.044556 |
| No prior hotel reference | 32,346 | 0.456792 | 0.456792 | 0.000000 |
| **Overall** | **223,074** | **0.421720** | **0.598007** | **+0.176287** |

Interpretation:

1. **Immediate continuation:** The label equals the most recently referenced hotel by taxonomy. A sufficiently large boost places that hotel first (when in the candidate set), so MRR=1 is largely built into the rule and segment definition. It does not demonstrate rich understanding of preferences.
2. **Return to earlier hotel:** The boost prioritizes a different, more recently referenced hotel; test MRR declines by 0.137275 compared with position only.
3. **New hotel after exploration:** The target was not in the retained hotel history; boosting a different hotel can push the target down. Test MRR declines by 0.044556.
4. **No prior hotel reference:** No behavioral boost is applied; the hybrid equals position only.

The selected hybrid's overall gain is **not** a uniform improvement across traveler behaviors. It exploits a frequent, highly predictable segment while degrading two others. There is no evidence here of conversion or booking lift.

## Relation to previous experiments

The selected hybrid test tie-neutral MRR (0.598007) matches the earlier last-item test MRR **with candidate-order tie-breaking** (0.598007), while last-item's independently tie-neutral test MRR was 0.504871. The explicit hybrid scores candidate position rather than silently using logged rank order to resolve ties. This numerical equality is consistent with the underlying ranking behavior, but is not proof of any new latent traveler signal.

## Evidence / interpretation / alternatives / follow-up

| Evidence | Interpretation | Limitation or alternative | Open question |
|---|---|---|---|
| Boosts 2 and 4 tie exactly | Ranking saturates with dominant last-hotel boost | No confidence interval; identical deterministic rankings under specified scoring rule | Should simpler lexicographic 'last item first, otherwise position' replace an arbitrary weight? |
| Immediate MRR = 1 | Last hotel is predictive in immediate continuation | Segment is defined by matching last hotel; not a separate generalization test | How should an online prediction point be defined before future event time? |
| Return / new-hotel segments decline | Unconditional last-hotel priority can hurt | Segment labels are ex post analysis, not necessarily known at inference time | Can a causal gating signal identify when to trust the latest hotel? |
| No-hotel segment unchanged | Hotel-ID behavioral boost provides no information | Search/filter context not included here | Which context baselines deserve a fair trial? |

## Caveats and reproducibility

- `impressions` and candidate order are recorded on **target click-out rows**. Pre-click availability and exposure are unverified; this is a reconstructed candidate-set benchmark.
- Histories contain retained 1–50 earlier events, not the full session. `search for item` is provisionally treated as hotel-specific; its raw metadata membership was audited separately.
- Position is an upstream-ranker signal, not a pure traveler-representation feature.
- Several examples can come from one session; no session-clustered uncertainty estimates have been computed.
- The reported results are for the observed six-day Trivago window, not proven stable across seasons or production Expedia traffic.
- Source: measured `data/processed/hybrid_position_v023.json`; preserve the JSON or a reproducible command/output record with the experiment, subject to repository data policies.

## Interview explanation

> I tested an explicit hybrid of prior candidate position and a fixed last-hotel boost. Validation selected a boost of two, but two and four yielded identical ranking metrics because the boost dominates the normalized position range. Test MRR increased from 0.422 to 0.598; however, nearly all the value came from immediate-continuation cases, which the rule predicts by construction. It reduced MRR on both returns to older hotels and selections of new hotels. I therefore treated this as a strong but uneven candidate-reranking baseline and documented the failure modes before considering a more adaptive method.

**Checkpoint conclusion:** Document and retain this baseline; do not infer that fixed-boost reranking solves the broader traveler-representation problem. Keep additional modeling paused pending a deliberate next hypothesis.
