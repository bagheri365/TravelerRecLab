# ADR-021 — Accept v0.3 mean pooling as a learned control, not a GRU performance target yet

**Status:** Accepted for documenting the reported run; next-model comparison gate remains open.

## Context

The first trainable model used hotel-ID embeddings, mean pooling of historical hotel events, candidate embedding similarity with bias, and click-out candidate-softmax training. The full run used 823,338 training examples and selected the strongest observed validation epoch (epoch 3 of 3). It reached validation MRR 0.467356 and test tie-neutral MRR 0.466170 on 223,074 examples. Last-item reached 0.504871 on the same test split in earlier evaluation. The position-plus-last-item hybrid reached 0.598007 but uses an additional logged-position signal.

## Decision

1. Treat v0.3 as a **successful implementation and measurement of a first learned non-sequential control**, not evidence that mean pooling beats heuristics or that GRUs are necessary.
2. Preserve the exact dataset inclusion rules, chronological split, train-only hotel vocabulary, candidate-softmax task and tie-neutral metrics in future architectural comparisons.
3. Distinguish feature-matched history-only comparison (mean pooling versus GRU) from ranking comparisons involving logged candidate position.
4. Flag training-budget adequacy as unresolved: validation MRR improved in all three reported epochs. Choose future training durations and checkpoints with validation, not with observed test performance.
5. Require behavioral-segment results and vocabulary coverage diagnostics before interpreting the meaning of any new model's aggregate MRR.

## Alternatives considered

- **Proceed straight to GRU and compare aggregate MRR only:** faster, but does not isolate order modeling and can reward training-budget differences.
- **Tune mean pooling against the test set:** rejected because it biases the final comparison.
- **Introduce metadata and context simultaneously with GRU:** rejected for the first architectural comparison; feature changes would confound the GRU effect.
- **Treat the position hybrid as the neural model's main target:** useful as an external reranking reference, but not input-feature-equivalent.

## Consequences and limitations

A GRU experiment should only support an order-modeling conclusion when feature input, candidate scoring, loss, dataset, and evaluation are matched, and the mean-pooling baseline has a reasonable validation-selected training budget. The candidate lists are from target click-out rows, not independently confirmed pre-click exposures. Click-out labels are not bookings. Examples from the same session are correlated. The current mean-pooling experiment has no four-pattern model metrics or replicate-seed uncertainty assessment. Unknown-ID fallback is not content-based cold-start generalization.

## Evidence and follow-up

Evidence: the project operator's v0.3 smoke and full training terminal logs; findings documented in `docs/experiments/v0_3_mean_pool_findings.md`. This ADR records the decision to *pause claims*, not to delay future experiments indefinitely. Review the full metrics artifact next, then define the specific controls for the first GRU run.
