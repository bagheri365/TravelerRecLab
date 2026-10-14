# Interview study notes — What v0.2 actually taught us

Use these as short answer drills, **not** as claims that a GRU has already been validated.

## 1. Why start with popularity and last-item instead of a GRU?

A baseline measures whether complex sequence learning is necessary. We observed high target repetition and short histories. Last-item reached **0.5049 tie-neutral test MRR** and remained our strongest simple baseline. Complexity should earn its operational cost.

## 2. What was wrong with the original 0.598 MRR?

When several hotels received tied scores, the original candidate order decided their ranks. This made the scoring model appear stronger than its discrimination justified. Neutralizing those ties lowered last-item MRR to **0.5049**. Tie-neutral metrics still cannot eliminate exposure bias.

## 3. How can last-item be excellent and poor at the same time?

It achieved **0.9092 MRR when the clicked hotel already appeared in the history** but only **0.1521 when it had not**. Overall MRR averages these very different populations. In the latter segment, hotel context and existing result ordering contain information that hotel-ID repetition scores lack.

## 4. Why does Recall@5 tell a different story?

On seen targets, five-minute recency scores achieved **0.9954 Recall@5**, above last-item's **0.9168**, despite lower MRR. Aggregating several previously viewed hotels increases the probability that the clicked one is included in the top five, whereas last-item is optimized by construction for a single recent favorite.

## 5. How would I fairly demonstrate that a GRU is better?

Compare GRU against a **tuned recency/action baseline and last-item**, with the *same* split, candidates, feature access, evaluation code, and reporting slices. Add a context/position-aware simple baseline if the GRU sees context. Measure quality improvements on unseen-in-history cases and the incremental serving cost. We have **not** run that experiment yet.

## 6. Why isn't a click-out a booking label?

A Trivago click-out indicates selecting an offer to leave for a booking partner. Whether a reservation completed is unobserved. Thus our benchmark teaches event modeling and candidate ranking, but booking propensity requires a different supervised data source.

## 7. What about users with no events?

Our sequence generator emitted **1,276,216** examples from **1,586,586** click-outs. Nearly all excluded click-outs lacked sufficient prior session history under the audited rules. This subset is useful for testing sequence models, but does not reflect all production requests. Production should support a context-only or nonpersonalized fallback and separately measure zero-history performance.

## Questions to answer verbally before the next milestone

- Precisely define `target_seen` versus an item that is new to the catalog.
- Explain why an expected rank under ties is more attributable to the model than logged-list tie-breaking.
- Explain why the best 5-minute decay is selected on validation, not test.
- Describe a fair GRU-vs-pooling experiment where sequence order is the *only intended architecture difference*.
- Explain which quality and latency metrics would justify a more complex online traveler encoder.

## Current checkpoint

**Established:** short-session behavior and repeat-interest signals are powerful, candidate position predicts click-outs, tied scores need careful handling, and aggregation helps top-five inclusion.

**Not established:** benefits of GRUs/Transformers, online booking lift, cold-start coverage, and a production-valid historical exposure timeline.
