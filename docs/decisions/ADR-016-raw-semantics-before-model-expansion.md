# ADR-016 — Audit raw semantics before expanding baselines

**Status:** Accepted as an audit plan; empirical claims pending raw-data execution.

## Context

Checkpoint A.2 measured four retained-history patterns across 1,276,216 emitted examples. Critical interpretations depend on whether `search for item` identifies a hotel, whether prior clickouts are fully represented, and whether history length 50 indicates true truncation.

## Decision

Freeze new modeling. Add a **read-only**, non-serving raw CSV audit that (a) checks `search for item` references against hotel metadata, (b) partitions raw clickouts into first/repeat/switch using full prior session histories, and (c) separates exactly-50 from >50 preceding-event histories and detects seen targets lost through truncation. Preserve the builder's provisional `(timestamp, step, source row)` ordering. Do not change the current examples, baselines, or v0.2 leaderboard.

## Alternatives rejected for now

- **Infer statistics from the capped JSONL:** cannot determine events discarded before the last 50.
- **Treat metadata membership as definitive proof:** ID overlap alone does not prove action meaning.
- **Retrain more complex models immediately:** would compound unresolved assumptions.

## Risks and validation

Full CSV loading and sorting can require significant RAM; use `--max-rows` only as a prefix smoke test, never for population proportions. Tests cover classification, ordering, eligibility and truncation. Compare `eligible_min_history_1` to the existing 1,276,216 examples and investigate discrepancies before accepting the full-data report. Treat original clickout-row candidates as a reconstruction only.
