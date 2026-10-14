import json

import pytest

from travelerreclab.data.eda_sequences import analyze_examples, write_report


def ex(user, session, ts, step, target, history):
    return dict(user_id=user, session_id=session, target_timestamp=ts,
                target_step=step, target_item=target, candidate_items=[target, 'other'],
                history=history)


def ev(ts, step, action, reference):
    return dict(timestamp=ts, step=step, action_type=action, reference=reference)


def test_eda_counts_repeats_and_fold_overlap(tmp_path):
    # November 5 UTC boundary, include two sessions for same traveler.
    boundary = 1541376000
    records = [
        ex('u', 's1', boundary - 10, 3, 'a', [ev(boundary - 30, 1, 'search for destination', 'Paris'),
                                               ev(boundary - 20, 2, 'interaction item image', 'a')]),
        ex('u', 's2', boundary + 10, 1, 'b', []),
        ex('v', 's3', boundary + 20, 2, 'c', [ev(boundary + 10, 1, 'clickout item', 'c')]),
    ]
    source = tmp_path / 'events.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in records))
    result = analyze_examples(source)
    assert result['examples'] == 3
    assert result['history_length_emitted']['p50'] == 1
    assert result['zero_history_examples'] == 1
    assert result['repeat_counts']['any_previous_item_equals_target'] == 2
    assert result['repeat_counts']['last_item_equals_target'] == 2
    assert result['repeat_counts']['previous_clickout_equals_target'] == 1
    assert result['candidate_coverage_emitted'] == 1
    assert result['identities_crossing_split']['users'] == 1
    assert result['identities_crossing_split']['sessions'] == 0
    write_report(result, tmp_path / 'out')
    assert (tmp_path / 'out' / 'summary.json').exists()
    assert 'Limitations' in (tmp_path / 'out' / 'report.md').read_text()


def test_order_anomaly_and_invalid_record(tmp_path):
    source = tmp_path / 'events.jsonl'
    source.write_text(json.dumps(ex('u', 's', 3, 3, 'x', [
        ev(2, 2, 'interaction item image', 'x'),
        ev(1, 1, 'interaction item image', 'y'),
        ev(3, 3, 'interaction item info', 'x')])) + '\n')
    result = analyze_examples(source)
    assert result['anomalies']['nonmonotonic_emitted_history'] == 1
    assert result['anomalies']['history_not_before_target_by_key'] == 1
    source.write_text('{broken json}\n')
    with pytest.raises(ValueError, match='line 1'):
        analyze_examples(source)


def test_empty_file_rejected(tmp_path):
    p = tmp_path / 'empty.jsonl'
    p.write_text('')
    with pytest.raises(ValueError, match='No examples'):
        analyze_examples(p)
