# ADR-008: Scope the first benchmark using observed data

**Status:** Provisional (pending targeted sequence EDA)
**Date:** 2026-10-13

## Question

Should TravelerRecLab train a general long-term traveler embedding or begin with
short-term click-out prediction on Trivago sessions?

## Evidence (user-run raw EDA)

- 15,932,992 events; 730,803 users; 910,683 (user, session) pairs.
- Observed data covers only November 1–6, 2018 (UTC).
- Median 4 events/session; p90 41; p99 212.
- Median 1 session/user; p90 2; p99 4.
- 1,586,586 click-outs, with 1,585,837 click targets appearing in their recorded
  candidate lists (approximately 99.95%).
- Impressions are attached to click-out rows; they are **not independently
  timestamped pre-click impression events**.

## Options

1. **Session-first click-out retrieval / offline candidate ranking:** tractable
   labels, observable event histories; limited long-term generalization.
2. **Long-term traveler model immediately:** closer to Expedia's target system,
   but observational window is only six days and most IDs have one session.
3. **Booking propensity:** cannot be evaluated on Trivago because the labels
   are partner click-outs, not confirmed bookings.

## Provisional decision

Build session-first next-click-out learning as an initial reproducible benchmark.
Leave long-term history and online-serving experiments on the roadmap, but do not
claim Trivago validates months-long preferences or booking lift.

## Required follow-up evidence

- Fraction of eligible click-outs with nonempty causal prefixes.
- Repeat-item and last-item target rates.
- Distribution of training-prefix lengths after truncation.
- Raw event ordering and step/timestamp duplicate validation.
- Chronological evaluation fold counts and identity overlap.

`eda_sequences.py` measures **emitted-example** statistics. It cannot recover
examples skipped by `build_sequences.py`, or prove ordering in the raw CSV. A
later raw audit must fill this gap; mark it **open**, not silently resolved.

## Interview answer (60 seconds)

> I scoped the initial experiment to click-out prediction because Trivago
> contains actual session interactions and associated candidate sets. Its
> six-day horizon and median one session per user cannot establish stable
> long-term preferences. I will compare strong recency-aware baselines with
> GRUs on a fixed temporal protocol and only increase complexity if quality
> gains justify the serving costs. I will not call click-outs bookings or
> assume click-out-row impression lists were observed at an earlier time.
