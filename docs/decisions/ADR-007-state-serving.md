# ADR-007 — Incremental session state serving

**Status:** Planned
**Date:** 2026-10-13

## Problem
Reflect new traveler actions quickly without training a new model for every event.

## Options
- Recompute pooled history on request: straightforward but increasingly expensive.
- **GRU state update per new event:** compact incremental update; order-sensitive and version-sensitive.
- Transformer re-encode bounded sequence or maintain cache: expressive but more operational state.

## Proposed experiment, not yet implemented
Replay ordered events; compute GRU hidden states incrementally and with a full batch pass using identical model weights and preprocessing. Compare values within numerical tolerance. Then measure update freshness and p50/p95/p99 latency. Tag stored states by model version, implement deduplication and event ordering. Plan for state rebuild on model changes.

## Revisit if
Pooling performs comparably with lower cost, state migration proves too fragile, or Transformer delivers material quality improvements.

## Interview explanation (60 seconds)
> Near-real-time embedding updates mean serving inference on new actions, not retraining network parameters. A GRU permits cheap recurrent state updates, but operational correctness requires event ordering, replay, deduplication, and matching model-state versions.
