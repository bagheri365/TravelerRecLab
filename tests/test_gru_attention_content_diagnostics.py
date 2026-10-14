"""Synthetic evaluation checks for content-GRU behavioral/coverage slicing."""
import csv
import json

import pytest

from travelerreclab.models.gru_attention_content import create_model, load_properties
from travelerreclab.models.gru_attention_content_diagnostics import (
    reciprocal_metrics, report_slices, target_coverage,
)


def test_coverage_partition():
    vocab = {'A': 2, 'B': 3}
    assert target_coverage('A', vocab, {'A'}) == 'clicked_in_train'
    assert target_coverage('B', vocab, {'A'}) == 'in_vocab_not_clicked_in_train'
    assert target_coverage('C', vocab, {'A'}) == 'outside_train_vocabulary'


def test_tie_neutral_rank():
    mrr, recall, ndcg = reciprocal_metrics([2., 2., 1.], 1)
    assert mrr == pytest.approx(.75)
    assert recall == 1
    assert ndcg == pytest.approx((1 + 1 / __import__('math').log2(3)) / 2)


def test_reject_unknown_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    ckpt = tmp_path / 'wrong.pt'
    torch.save({'architecture': 'other'}, ckpt)
    with pytest.raises(ValueError, match='architecture'):
        report_slices(ckpt, tmp_path / 'none', tmp_path / 'none', 'validation')


def test_synthetic_report_reproduces_counts_and_metrics(tmp_path):
    torch = pytest.importorskip('torch')
    data = tmp_path / 'seq.jsonl'
    def example(time, target, candidates, history):
        return {'target_timestamp': time, 'target_item': target,
                'candidate_items': candidates, 'history': history}
    def history(h):
        return [{'action_type': 'interaction item image', 'reference': h}]
    rows = [
        example(1541030400, 'A', ['A', 'B'], history('A')),
        example(1541030401, 'A', ['A', 'B'], history('A')),
        example(1541376000, 'A', ['A', 'B'], history('A')),
        example(1541376001, 'B', ['A', 'B'], history('A')),
        example(1541376002, 'C', ['A', 'C'], history('A')),
    ]
    data.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    meta = tmp_path / 'metadata.csv'
    with meta.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['item_id', 'properties'])
        writer.writeheader()
        for h in ('A', 'B', 'C'):
            writer.writerow({'item_id': h, 'properties': 'WiFi'})
    _, tags = load_properties(meta)
    vocab = {'A': 2, 'B': 3}
    model = create_model(4, len(tags), 4)
    ckpt = tmp_path / 'model.pt'
    torch.save({'architecture': 'id_plus_property_attention_gru', 'vocabulary': vocab,
                'tag_vocabulary': tags, 'dim': 4, 'content_weight': 0.5,
                'model': model.state_dict()}, ckpt)
    result = report_slices(ckpt, data, meta, 'validation', batch_size=2)
    metrics = result['metrics']
    assert metrics['all']['n'] == 3
    assert metrics['behavior/immediate_continuation']['n'] == 1
    assert metrics['behavior/new_hotel_after_exploration']['n'] == 2
    assert metrics['train_target/clicked_in_train']['n'] == 1
    assert metrics['train_target/in_vocab_not_clicked_in_train']['n'] == 1
    assert metrics['train_target/outside_train_vocabulary']['n'] == 1
    assert metrics['properties/with_properties']['n'] == 3
    assert 0 <= metrics['all']['mrr_neutral_ties'] <= 1
    assert sum(metrics[f'train_target/{s}']['n'] for s in (
        'clicked_in_train', 'in_vocab_not_clicked_in_train',
        'outside_train_vocabulary')) == metrics['all']['n']
