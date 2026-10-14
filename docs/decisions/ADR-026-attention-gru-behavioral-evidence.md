# ADR-026 — Interpret candidate-aware GRU results by behavior, not aggregate MRR alone

**Status:** Accepted for current offline benchmark; no production deployment decision.

## Context

v0.3.1 GRU achieved test tie-neutral MRR 0.515058 but only 0.290263 on `return_to_earlier_hotel`. Its original-order vs shuffled-history diagnostic indicated strong preference for immediate continuation and poorer results on earlier returns. The v0.4 candidate-aware GRU can query earlier recurrent states for each candidate and retains a final-state skip. The preregistered question was whether this helps earlier returns without reducing immediate continuation.

See [`../experiments/v0_4_attention_gru_findings.md`](../experiments/v0_4_attention_gru_findings.md) for the full validation and test MRR / Recall@5 / NDCG@5 matrices, reproducibility commands and provenance.

## Evidence and decision

- Full test set (223,074 examples): aggregate MRR **0.515058 → 0.520126** (+0.005067); Recall@5 **0.603734 → 0.613428**; NDCG@5 **0.510856 → 0.517561**.
- Return-to-earlier-hotel (10,940 examples): MRR **0.290263 → 0.359127** (+0.068864, approximately 23.7% relative); Recall@5 **0.524625 → 0.678621**.
- Immediate continuation (93,005 examples): MRR **0.947021 → 0.949408**; no regression by this point estimate.
- Validation replicates the directional earlier-return result (**0.294383 → 0.367959**), while new-hotel-after-exploration validation MRR slightly declines.
- Roughly **two-thirds** of the total test MRR gain is attributable arithmetically to the return-to-earlier-hotel group, despite the group comprising ~4.9% of examples.

**Decision:** Mark the targeted behavioral hypothesis *supported by descriptive offline measurements*. Preserve v0.4 as a benchmarked learned-encoder variant. Do **not** claim candidate attention *alone* caused the gain, statistical significance, or production superiority.

## Alternatives considered

1. **Keep the final-state-only GRU:** simpler and already strong on continuation, but materially weaker on returns in these reports.
2. **Replace GRU with Transformer immediately:** architectural cost and mechanisms increase without an isolated diagnosis of the remaining new-to-history weakness.
3. **Add candidate position to v0.4:** could lift offline MRR but would confound the hotel-ID-only architecture comparison and relies on target-row candidate ordering whose pre-click serving availability is unverified.
4. **Introduce candidate metadata/search context in a separate experiment:** promising direction for new-to-history hotel generalization; requires its own data provenance, feature availability, and matched evaluation contract.

## Consequences and follow-up controls

- Continue to report **segment MRR, Recall@5 and NDCG@5** alongside aggregate ranking metrics, with the exact group populations.
- Preserve train-only vocabulary and chronological splits; checkpoint selection must use validation only. Treat existing test breakdowns as frozen post-hoc evidence, not a tuning target.
- For stronger architectural attribution, separately test skip-only, attention-without-skip, and parameter-matched GRU variants on validation, with more than one seed and session-clustered uncertainty where practical.
- For the next genuinely new task, investigate hotel attributes or search-context signals for **new-to-history candidate ranking**, with explicitly verified feature availability and no target leakage.
- No deployment, booking uplift, latency, convergence, cold-start, or causal claims follow from this ADR.
