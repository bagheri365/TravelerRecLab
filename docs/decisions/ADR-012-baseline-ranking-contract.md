# ADR-012 — Freeze the first baseline-ranking contract

**Status:** Provisional; implementation complete, full-data results pending.

## Context

Our Trivago click-out examples contain causal histories and candidate lists attached to target rows. Approximately 46.7% of targets had appeared in earlier interactions. Models must be compared without accidentally learning from future click-outs or changing candidate sets.

## Decision

Use UTC target-time holdouts (train before November 5, validate November 5, test November 6), fit global popularity only on the training targets, score identical candidate lists with deterministic tie-breaking, and publish MRR / Recall@K / NDCG@K. Keep `min_history=1` as the initial comparison population while recognizing that this omits cold-start cases.

The first implementation includes SHA256 random scoring, click-out popularity, last hotel, action-weighted history, and exponentially decayed action-weighted history. Use their results as evidence, not proof that recurrence is unnecessary.

## Alternatives and trade-offs

- **Random prefix split:** easier and larger train/validation mixtures, but highly overlapping histories may leak across splits.
- **User-disjoint split:** evaluates unseen users, but differs from serving known-user histories and can be difficult with a six-day sample.
- **Full-catalog next-item retrieval:** avoids depending on recorded click-out candidates, but requires a separate negative/retrieval protocol.
- **Bookability-conditioned ranking:** closer to Expedia's production setting, but live availability and confirmed bookings aren't present.

## Unresolved / revise before making a production claim

1. Candidate lists are logged on target click-out rows, with no independent impression timestamp.
2. `search for item` references need explicit hotel-ID semantics verification.
3. Positional bias from deterministic ties needs a position-only baseline.
4. Session event ordering is provisional, with timestamp-first/step tie-break policy.
5. Real-time serving, full cold-start coverage, confidence intervals, and hyperparameter tuning remain future milestones.

## Interview answer

> I used a chronological split and kept every baseline on the same candidate ranking task. Popularity is learned only from the training period, while session-history features are built causally. I documented the exposure and cold-start limitations instead of claiming the benchmark directly measures booking conversion or production ranking quality.
