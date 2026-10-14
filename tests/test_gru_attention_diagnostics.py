"""v0.4 saved-checkpoint behavioral diagnostics."""
from datetime import datetime, timezone
import json

import pytest

from travelerreclab.models.gru_attention_diagnostics import analyze, run
from travelerreclab.models.gru_attention import create_model


def example(history, target, day=5):
    return {
        'target_timestamp': int(datetime(2018, 11, day, tzinfo=timezone.utc).timestamp()),
        'candidate_items': ['A', 'B', 'C'], 'target_item': target,
        'history': [{'action_type': 'interaction item image', 'reference': h}
                    for h in history],
    }


def test_four_segments_and_overall_count(tmp_path):
    torch = pytest.importorskip('torch')
    rows = [example(['A'], 'A'), example(['A', 'B'], 'A'),
            example(['A'], 'C'), example([], 'B')]
    path = tmp_path / 'examples.jsonl'
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    model = create_model(5, 8)
    metrics = analyze(model, path, {'A': 2, 'B': 3, 'C': 4}, split='validation', batch_size=3)
    assert metrics['all']['n'] == 4
    for group in ('immediate_continuation', 'return_to_earlier_hotel',
                  'new_hotel_after_exploration', 'no_prior_hotel_reference'):
        assert metrics[group]['n'] == 1
        assert 0 <= metrics[group]['mrr_neutral_ties'] <= 1


def test_checkpoint_roundtrip_and_aggregate_consistency(tmp_path):
    torch = pytest.importorskip('torch')
    model = create_model(5, 8)
    vocab = {'A': 2, 'B': 3, 'C': 4}
    checkpoint = tmp_path / 'attention.pt'
    torch.save({'model': model.state_dict(), 'vocabulary': vocab, 'dim': 8,
                'architecture': 'hotel_id_candidate_attention_gru'}, checkpoint)
    data = tmp_path / 'examples.jsonl'
    data.write_text(json.dumps(example(['A', 'B'], 'B')) + '\n')
    report = run(checkpoint, data, tmp_path / 'report.json', split='validation')
    assert report['metrics']['all']['n'] == 1
    assert report['metrics']['all']['mrr_neutral_ties'] == report['metrics']['immediate_continuation']['mrr_neutral_ties']
    assert json.loads((tmp_path / 'report.json').read_text())['metrics'] == report['metrics']


def test_rejects_wrong_architecture(tmp_path):
    torch = pytest.importorskip('torch')
    checkpoint = tmp_path / 'wrong.pt'
    torch.save({'architecture': 'hotel_id_gru'}, checkpoint)
    with pytest.raises(ValueError, match='Expected'):
        run(checkpoint, tmp_path / 'missing.jsonl', tmp_path / 'report.json')
