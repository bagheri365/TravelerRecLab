# ADR-014 — Freeze interpretation of baseline metrics and segmented evaluation

- **Status:** Accepted for reporting v0.2/v0.2.1; further modeling decisions remain provisional
- **Date:** 2026-10-13
- **Context:** Trivago next-click-out among recorded candidates
- **Evidence:** [`../experiments/v0_2_checkpoint_findings.md`](../experiments/v0_2_checkpoint_findings.md)

## Question

Why was initial last-item MRR ~0.598, and does it show that behavioral sequence modeling is unnecessary?

## Alternatives considered

1. Keep logged-order tie-breaking: reproducible but attributes inherited original-list order to models that assigned tied scores.
2. Use tie-neutral expected ranks: isolates score discrimination under ties, but **does not** address all exposure and selection biases.
3. Use position-only as the main comparison: tells us about the existing ranking but not incremental behavioral value.
4. Report aggregate metrics only: concise but hides the severe seen/unseen disparity.

## Decision

- Use **tie-neutral MRR** as the preferred offline comparison metric for these baseline scores, while retaining logged-order MRR as a diagnostic.
- **Always report seen-in-history and unseen-in-history target segments** with aggregate results.
- Report **Recall@5** as a complementary objective to MRR, especially for multi-item retrieval.
- Treat the **last-item baseline (~0.5049 neutral test MRR)** as the primary simple behavioral comparator; also report **5-minute recency (~0.4944)** and **position-only (~0.4217)**.
- Do not infer superior embedding quality, causal lift, or booking outcomes from these observations.

## Rationale and evidence

Original-candidate-order tie-breaking raised last-item test MRR from **0.5049** to **0.5980**. More importantly, last-item achieves **0.9092** MRR for previously seen targets and only **0.1521** for previously unseen targets. The position baseline achieves **0.3421** on unseen targets. Therefore aggregate results alone would drive us toward the wrong modeling priorities.

Five-minute recency weighting has **0.9954 Recall@5** on seen targets versus **0.9168** for last-item, showing that metric selection changes the apparent winner.

## Risks and limitations

- The recorded impression list may not correspond to independently logged pre-click exposure.
- “Target unseen” means unseen *in the user's available prefix*, not necessarily a cold-start item.
- The short six-day source window limits claims about stable long-term traveler interests.
- Differences are point estimates without uncertainty intervals.
- Baseline implementations and tie semantics should remain versioned and tested.

## Revisit when

A model or business metric changes the objective; candidate sets differ; independently timestamped impression logs become available; confidence intervals or rigorous offline policies indicate another evaluation design; or a GRU is compared fairly using equivalent contextual features.

## Interview response (~60 seconds)

> I first measured strong last-item performance, but found that tie-breaking inherited the logged candidate order. After neutralizing ties, its MRR moved from about 0.598 to 0.505. Segmenting the data showed 0.909 MRR for hotels previously engaged with, versus 0.152 for hotels not seen in the available history. That told me our simple model was a strong repeat-interest predictor, but not yet a general contextual hotel recommender. A position-only benchmark reached 0.342 on unseen targets. I would keep tie-neutral MRR and segmented reporting, build a fair contextual baseline, and only then test whether a GRU adds value beyond the same features and candidate sets.
