# Checkpoint A.1 — Understand a real Trivago prediction example

**Status:** study protocol; not a new experiment or modeling result.

## Purpose and success criteria

Before extending baselines or learning embeddings, trace an actual click-out from the
raw event schema into the existing next-click-out benchmark. Completion means we
can explain (1) the prediction time, (2) the exact permitted history, (3) the
label and candidate construction, (4) what is observed rather than inferred,
(5) any limitations. Do not change models during this checkpoint.

## What is established from the v0.1 sequence builder

- Raw columns include `user_id`, `session_id`, `timestamp`, `step`,
  `action_type`, `reference`, `platform`, `city`, `device`, `current_filters`,
  `impressions`, and `prices`.
- Session identity is `(user_id, session_id)`; events are ordered by
  `(timestamp, step, original input row)` within sessions.
- Each eligible `clickout item` produces one example. The target is the
  click-out event's `reference`, if nonempty and present in its `impressions`
  and candidate/prices arrays have equal, nonzero length.
- The `history` consists of earlier events from the same session, truncated
  to the most recent 50. By default at least one preceding event is required.
  Earlier click-outs may appear in history; the **current** target click-out
  does not.
- History events retain `timestamp`, `step`, `action_type`, `reference`.
  They do not retain all raw context columns. Do not claim a model uses city,
  device, filters, or hotel metadata merely because the raw CSV has them.
- The candidate items and prices are taken from **the target click-out row**.
  Their availability before the user's decision is **not established**.
  Treat this as a reconstructed candidate-set *offline benchmark*, not a
  validated pre-decision online ranking input.
- `reference` depends on `action_type`. A destination string or filter
  must not be treated as a hotel ID without verifying action semantics.

## Part 1: inspect the actual output example

Run locally from the repository root, after constructing `next_clickout.jsonl`.
This prints **one** example without user or session identifiers and without
outputting raw hotel IDs (they are replaced by H1, H2, ...). It prefers a
history where the label is not one of the previously referenced hotel IDs,
so we can scrutinize a difficult case. It makes no assumption that any
non-hotel action's `reference` is a hotel ID.

```bash
python - <<'PY'
import json
from pathlib import Path

hotel_actions = {
    'clickout item', 'interaction item image', 'interaction item info',
    'interaction item rating', 'interaction item deals', 'search for item',
}
path = Path('data/processed/next_clickout.jsonl')
chosen = None
with path.open(encoding='utf-8') as stream:
    for line in stream:
        ex = json.loads(line)
        past_hotels = {str(e['reference']) for e in ex['history']
                       if e['action_type'] in hotel_actions and e.get('reference')}
        if ex['target_item'] not in past_hotels and len(ex['history']) >= 3:
            chosen = ex
            break
if chosen is None:
    raise SystemExit('No suitable example found; inspect the first JSONL row instead.')

ids = {}
def hotel_name(item):
    item = str(item)
    if item not in ids:
        ids[item] = f'H{len(ids) + 1}'
    return ids[item]

for item in chosen['candidate_items']:
    hotel_name(item)
for e in chosen['history']:
    if e['action_type'] in hotel_actions and e.get('reference'):
        hotel_name(e['reference'])

print('Prediction timestamp:', chosen['target_timestamp'])
print('Target:', hotel_name(chosen['target_item']))
print('Candidate count:', len(chosen['candidate_items']))
print('Target candidate position:', chosen['candidate_items'].index(chosen['target_item']) + 1)
print('History event count (possibly truncated):', len(chosen['history']))
print('History:')
for e in chosen['history']:
    ref = (hotel_name(e['reference']) if e['action_type'] in hotel_actions
           and e.get('reference') else '[non-hotel/empty reference]')
    print(' ', e['timestamp'], e['step'], e['action_type'], ref)
print('Candidates:', [hotel_name(x) for x in chosen['candidate_items']])
PY
```

**Interpret carefully:** `search for item` is provisionally classified as a
hotel-referencing action for inspection. Verify this against actual raw rows
before deciding to use it as a model feature. The printed example is **real
local dataset content**, but hotel names H1... are anonymized placeholders.

## Part 2: annotate the example, without guessing

Fill this out *after* running the command:

| Question | Observed answer | Evidence / uncertainty |
|---|---|---|
| What was the target action and label? | TODO | |
| How many previous events, of which action types? | TODO | |
| Which references were hotel-specific? | TODO | |
| Was the target item seen previously? | TODO | |
| What was its position in the recorded candidate list? | TODO | |
| Which columns existed in raw data but disappeared from history? | TODO | |
| Was the candidate set demonstrably available before the click-out? | **No evidence** | Recorded on the target click-out row |
| Could the ordering policy affect this example? | TODO | Timestamp ties / repeated steps possible |

## Part 3: questions to carry forward

1. Is the training target an *immediate click-out*, or the first click-out
   following a specific search action? (Answer from the builder, not intuition.)
2. Could repetitive image interactions cause many near-duplicate history signals?
3. What population is excluded by `min_history=1` and candidate validity checks?
4. Does a good model predict *repeat selection*, *novel hotel selection*, or both?
5. Is the candidate list an observed exposure at prediction time or a
   retrospective evaluation convenience?

**Gate to A.2:** We can narrate one real example end-to-end, identify at least
one possible label/candidate validity limitation, and name which raw context
columns are currently absent from the history. No neural modeling until the
problem and baseline interpretation are defensible.
