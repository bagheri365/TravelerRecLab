# Interview drill — What the Trivago EDA taught us

Read with [the evidence ledger](../data/trivago_eda_findings_v0_1_8.md) and [ADR-011](../decisions/ADR-011-freeze-pre-baseline-data-contract.md).

| Interviewer asks | Answer in one breath | Follow-up risk |
|---|---|---|
| Why next click-out? | It's the strongest observed hotel-selection signal in this dataset. | Not a booking or true booking propensity label. |
| Why start with simple baselines? | ~46.7% of emitted targets appeared in prior history, so last-item and recency may already work well. | Diagnostic must distinguish hotel-valued references from other `reference` types. |
| What event features? | Typed action/reference, destination/context, time gaps, prior interactions; candidate hotel/price at scoring time. | Some context fields are absent by action design, not random missingness. |
| Why sequence length 50? | Initial cutoff; p90 emitted prefix length 44, but p99=50 is truncated. | Run 20/50/100 ablation; don't call 50 the natural p99. |
| Why not rely on user ID? | Median user has only one observed session and anonymous users need fallbacks. | Session model is not a complete long-term traveler model. |
| How prevent leakage? | Build strictly preceding histories and chronological target-time splits; train popularity only on training outcomes. | Candidate impressions are logged with click-out, not verified prior exposures. |
| How order events? | Deterministic timestamp, step, source-row tie-breaker for benchmark; audit conflicts and alternatives. | Several sessions have repeated step 1 hours/days apart. |
| What about cold-start? | Evaluate zero-history separately and define explicit context/popularity fallback. | Emitted training population covers only ~80.4% of raw click-outs. |
| Are metadata embeddings justified? | Coverage is ~99.93% of distinct clicked IDs; useful candidate for cold-start ablation. | Candidate coverage and historical availability are not established. |
| Why MRR? | One positive clicked hotel per reconstructed candidate list; MRR rewards ranking it early. | Also report Recall/NDCG, coverage and slice results. |

## Questions to answer with data in v0.2

- Does last valid hotel really beat global popularity under identical candidate sets?
- Is a simple action/recency score better after the model sees only legal historical hotel references?
- How do results differ for repeated vs new-to-session targets, short vs long prefixes, and unseen training hotels?
- How sensitive are results to the chronological split and ordering anomalies?
- Does any model offer enough improvement to justify learned embeddings and online state management?

**Important:** Do not claim baseline or GRU improvements until experiments are run.
