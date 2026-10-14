# ADR-027 — Audit hotel content coverage before designing a content-aware encoder

**Status:** Accepted for audit protocol; no model selection yet.

## Context

The v0.4 candidate-attention GRU improved earlier-hotel return MRR (test 0.2903 → 0.3591) but not new-hotel-after-exploration MRR (~0.1720). Hotel ID embeddings cannot describe unseen IDs except through shared UNK. The Trivago dataset includes `item_metadata.csv` with `item_id` and `properties` tags, but we have not quantified usable property coverage for our **eligible prediction population**.

## Decision

Audit join coverage and tag availability *before* training a content encoder. Separate (a) new relative to **retained session history**, (b) absent from **train candidate/history vocabulary**, and (c) never a **positive train target**. Measure exact population and candidate-slot denominators; do not confuse metadata availability with predictive utility or point-in-time serving availability.

## Alternatives considered

- Train an ID+content encoder immediately: rejected for now; the useful-feature and join-population assumptions are unverified.
- Switch to a Transformer: does not directly address the unseen-ID problem.
- Use logged position: powerful but independent of hotel-content generalization and its online availability remains unverified.

## Acceptance / revisit

Accept the audit when it runs on full data and reports coverage by split and behavior. Decide the next modeling experiment from observed coverage rather than expectations. Keep original GRU and attention GRU as reference checkpoints; do not retune them using test-set findings.
