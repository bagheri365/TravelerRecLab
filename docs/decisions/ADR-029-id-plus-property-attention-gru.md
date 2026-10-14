# ADR-029 — Add hotel property content to attention-GRU as a separate experiment

**Status:** Proposed for controlled v0.5.1 experiment.

## Context

The v0.4 attention-GRU increased earlier-hotel return MRR from 0.290263 to
0.359127 but left new-hotel-after-exploration MRR at 0.172012. The v0.5 audit
found nearly complete metadata coverage, while 98.15% of new-to-session
clicked targets already belong to the train ID vocabulary. This is not
primarily an ID-UNK coverage problem.

## Decision

Keep the v0.4 history model, candidate-attention mechanism, chronological
splits, and candidate-softmax objective. Inject mean property-tag vectors
into **both history hotel and candidate** representations. Add a tag-only
candidate prior to distinguish properties even without hotel-specific history.
Use a preset content coefficient of 0.5 for this initial run, without test
optimization. Use exact raw hotel IDs for metadata lookup to avoid collapsing
all train-unknown hotels before content is encoded.

## Alternatives and tradeoffs

- More attention layers or a Transformer: not motivated by the dominant
  new-to-session weakness as directly as candidate content.
- Candidate position: strong, but not confirmed as available pre-click and
  would confound the representation-only comparison.
- Property tags only: useful later as a no-ID ablation for cold start, but
  removes known-ID information from this first experiment.
- Add prices, locations, or search context: potentially stronger for unseen
  candidate ranking, but a separate feature and data-time audit is needed.

## Exit criteria and constraints

Record full and four-segment metrics, especially new-to-history and
train-vocabulary-missing slices; examine continuity and older-hotel returns.
The tag-only prior makes v0.5.1 a **feature and scoring extension**, not an
isolated test of candidate-attention architecture. Do not assert point-in-time
correctness, statistical significance, or online conversion gain. A finding
is accepted only after validation-selected training and transparent testing.
