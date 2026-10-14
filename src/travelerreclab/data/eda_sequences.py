"""Streaming audit of v0.1 next-click-out JSONL examples.

This diagnoses *emitted examples*, not excluded click-outs or raw CSV ordering.
The latter require a separate source-log audit before leakage safety is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path


def _quantile(hist: Counter[int], fraction: float) -> int | None:
    total = sum(hist.values())
    if not total:
        return None
    threshold = max(1, int(total * fraction + 0.999999))
    running = 0
    for value, count in sorted(hist.items()):
        running += count
        if running >= threshold:
            return value
    return None


def analyze_examples(path: str | Path, *, split_day: str = '2018-11-05') -> dict:
    """Read a JSONL file incrementally; keep aggregate counters and identity sets."""
    boundary = int(datetime.strptime(split_day, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    n = 0
    lengths: Counter[int] = Counter()
    repeat = Counter()
    anomalies = Counter()
    by_day = Counter()
    user_days: dict[str, int] = defaultdict(int)
    session_days: dict[tuple[str, str], int] = defaultdict(int)
    target_in_candidates = 0
    for line_no, line in enumerate(Path(path).open(encoding='utf-8'), 1):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            history = ex['history']
            target = str(ex['target_item'])
            ts = int(ex['target_timestamp'])
            step = int(ex['target_step'])
            user, session = str(ex['user_id']), str(ex['session_id'])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'Invalid JSONL record at line {line_no}: {exc}') from exc
        n += 1
        lengths[len(history)] += 1
        day = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d')
        by_day[day] += 1
        fold = 1 if ts < boundary else 2
        user_days[user] |= fold
        session_days[(user, session)] |= fold
        if target in {str(item) for item in ex.get('candidate_items', [])}:
            target_in_candidates += 1
        preceding = []
        previous_key = None
        for event in history:
            key = (int(event['timestamp']), int(event['step']))
            if previous_key is not None and key < previous_key:
                anomalies['nonmonotonic_emitted_history'] += 1
            if key >= (ts, step):
                anomalies['history_not_before_target_by_key'] += 1
            preceding.append(event)
            previous_key = key
        if not preceding:
            repeat['empty_history'] += 1
        last = preceding[-1] if preceding else None
        if last and str(last.get('reference')) == target and str(last.get('action_type')) in {
            'interaction item image', 'interaction item info', 'interaction item rating',
            'interaction item deals', 'search for item', 'clickout item'
        }:
            repeat['last_item_equals_target'] += 1
        if any(str(e.get('reference')) == target and str(e.get('action_type')) in {
            'interaction item image', 'interaction item info', 'interaction item rating',
            'interaction item deals', 'search for item', 'clickout item'
        } for e in preceding):
            repeat['any_previous_item_equals_target'] += 1
        if any(str(e.get('reference')) == target and e.get('action_type') == 'clickout item'
               for e in preceding):
            repeat['previous_clickout_equals_target'] += 1
    if n == 0:
        raise ValueError('No examples found')
    return {
        'source': str(path), 'examples': n, 'split_day_utc': split_day,
        'history_length_emitted': {'p50': _quantile(lengths, .5),
                                   'p90': _quantile(lengths, .9),
                                   'p99': _quantile(lengths, .99),
                                   'max': max(lengths)},
        'zero_history_examples': repeat['empty_history'],
        'repeat_counts': dict(repeat),
        'repeat_rates': {k: round(repeat[k] / n, 6) for k in (
            'last_item_equals_target', 'any_previous_item_equals_target',
            'previous_clickout_equals_target')},
        'candidate_coverage_emitted': round(target_in_candidates / n, 6),
        'anomalies': dict(anomalies),
        'examples_by_day_utc': dict(sorted(by_day.items())),
        'identities_crossing_split': {
            'users': sum(v == 3 for v in user_days.values()),
            'sessions': sum(v == 3 for v in session_days.values()),
            'total_users': len(user_days), 'total_sessions': len(session_days)
        },
        'limitations': [
            'Only examples admitted by v0.1 are counted; excluded click-outs are not audited.',
            'History lengths are capped by --max-history at sequence-generation time.',
            'History ordering here checks sorted output, NOT ordering or duplicate steps in raw CSV.',
            'Timestamp ties and unseen earlier rows cannot be proven safe from emitted JSONL alone.',
            'Split overlap is based on emitted targets; user overlap is not itself leakage.',
            'Candidate sets are logged on click-out rows, not verified pre-click impressions.'
        ]
    }


def write_report(summary: dict, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Targeted sequence EDA — observed results', '',
        '> Generated from v0.1 JSONL training examples; not the full raw event log.', '',
        f"- Emitted training examples: **{summary['examples']:,}**",
        f"- Emitted history-length quantiles: `{summary['history_length_emitted']}`",
        f"- Candidate coverage among emitted examples: **{summary['candidate_coverage_emitted']:.2%}**",
        f"- Repeated-target rates: `{summary['repeat_rates']}`",
        f"- Example-level ordering anomalies: `{summary['anomalies']}`", '',
        '## Target counts by UTC day', '',
        *[f'- {k}: {v:,}' for k, v in summary['examples_by_day_utc'].items()], '',
        f"## Split diagnostic: before {summary['split_day_utc']} UTC versus on/after", '',
        f"- Identity overlap: `{summary['identities_crossing_split']}`", '',
        '## Limitations', '',
        *[f'- {x}' for x in summary['limitations']], '',
        '## Decisions to revisit', '',
        '1. Is `max_history=50` sufficient? Re-run generator with 100 and compare uncapped histories.',
        '2. How many click-outs were excluded (including first events)? Compare with raw audit totals.',
        '3. Validate raw CSV step uniqueness, timestamp reversals and ties before asserting causality.',
        '4. Specify event-time split and online-available-time assumptions; do not infer them from user overlap.',
        '5. Compare last-item, repeat-item and recency models on a frozen evaluation protocol.', ''
    ]
    (path / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/processed/next_clickout.jsonl')
    parser.add_argument('--output-dir', default='data/processed/sequence_eda')
    parser.add_argument('--split-day', default='2018-11-05')
    args = parser.parse_args()
    summary = analyze_examples(args.input, split_day=args.split_day)
    write_report(summary, args.output_dir)
    print(f"Analyzed {summary['examples']:,} emitted examples; see {args.output_dir}/report.md")


if __name__ == '__main__':
    main()
