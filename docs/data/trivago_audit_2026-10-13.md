# Trivago `train.csv` — observed data audit

**Source:** User-provided stdout from `python -m travelerreclab.data.inspect_trivago` on 2026-10-13.
**Status:** Descriptive; no independent execution against the full dataset in this repository patch.

## Shape and coverage

| Statistic | Observed |
|---|---:|
| Events | 15,932,992 |
| Columns | 12 |
| Unique `user_id` | 730,803 |
| Reported sessions | 910,683 |
| Session events mean | 17.50 |
| Session events median | 4 |
| Session events p90 | 41 |
| Session events p99 | 212 |
| Maximum events/session | 3,522 |

The inspection script reports session counts; confirm the grouping key is `(user_id, session_id)` before assuming it counts globally unique sessions.

## Actions

| `action_type` | Event count |
|---|---:|
| `interaction item image` | 11,860,750 |
| `clickout item` | 1,586,586 |
| `filter selection` | 695,917 |
| `search for destination` | 403,066 |
| `change of sort order` | 400,584 |
| `interaction item info` | 285,402 |
| `interaction item rating` | 217,246 |
| `interaction item deals` | 193,794 |
| `search for item` | 152,203 |
| `search for poi` | 137,444 |

## Missing fraction

| Field | Missing |
|---|---:|
| `current_filters` | 92.76% |
| `impressions` | 90.04% |
| `prices` | 90.04% |
| All other reported columns | 0% |

Those fields are event-type dependent; missingness is not necessarily a data quality fault. Do not backfill past events with future-row impressions or prices.

## What this does *not* establish

- Number of valid next-click-out training examples or fraction with non-empty history.
- Coverage of 50-event histories **at target timestamps**.
- Candidate-list membership rate or candidate-set ranking validity.
- Booking events, booking value, or booking conversion labels.
- Relative model quality. Those require experiments.

## Suggested follow-up measurements

1. Number of valid training rows, exclusions by reason, and click-outs with zero prior events.
2. History length percentiles at click-out targets.
3. Candidate-item membership, list size, and price alignment.
4. Duplicate events, tied timestamps/steps, repeated hotel interactions.
5. Sessions and users crossing chronological split boundaries.
