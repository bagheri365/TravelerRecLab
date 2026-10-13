# Interview decision drill: TravelerRecLab

Practice answering each in 45–60 seconds; avoid claiming unmeasured performance gains.

1. **Why Trivago?** Ordered, typed hotel behavior; no confirmed bookings. [ADR-001](../decisions/ADR-001-dataset-selection.md).
2. **Why next click-out, not next event?** Images dominate; click-out is closer to a hotel choice but is not a booking. [ADR-002](../decisions/ADR-002-prediction-target.md).
3. **How avoid leakage?** Strict historical prefixes and timestamp-aware evaluation; candidate lists are from the target row. [ADR-003](../decisions/ADR-003-causal-prefixes.md).
4. **Why 50 events?** Initial cost/truncation trade-off motivated by the overall session distribution; compare 20/50/100 at click-out time. [ADR-004](../decisions/ADR-004-sequence-length.md).
5. **Why mean/recency pooling first?** Strong, cheap control to isolate value of sequence order. [ADR-005](../decisions/ADR-005-evaluation-and-baselines.md).
6. **Why GRU vs Transformer?** Incremental state is attractive for streaming, but final selection depends on quality, freshness, latency, and lifecycle cost. [ADR-007](../decisions/ADR-007-state-serving.md).
7. **What is PAD vs OOV?** PAD is non-data and masked; OOV is a real unknown item, with a distinct representation and optional metadata features.
8. **How validate embeddings?** Hold out future events, compare candidate ranking to tuned pooling, and test on sparse and changing-intent segments; don't confuse offline click-out metrics with booking conversion.

## A reusable answer structure

**Constraint → options → decision → evidence → failure mode → experiment to reconsider.**

## Open interview questions

- Does evaluating against target-row impressions answer a prediction-time ranking question?
- How would booking propensity be trained with a different dataset and label definition?
- How should rare travelers, anonymous sessions, and returning travelers be represented?
- How is state rebuilt after a model version change or out-of-order events?
