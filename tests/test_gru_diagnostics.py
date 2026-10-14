import json

import pytest

torch = pytest.importorskip('torch')

from travelerreclab.models.gru import create_model
from travelerreclab.models.gru_diagnostics import (
    analyze, expected_ranking_metrics, pattern_for, shuffled_histories, run,
)


def row(hotels, target, ts=1541376000):
    return {'target_timestamp': ts, 'target_item': target,
            'candidate_items': ['A', 'B', 'C'],
            'history': [{'action_type': 'interaction item image', 'reference': h,
                         'timestamp': ts-1} for h in hotels] or
                       [{'action_type': 'search for destination', 'reference': 'City',
                         'timestamp': ts-1}]}


def test_patterns():
    assert pattern_for(row(['A', 'B'], 'B')) == 'immediate_continuation'
    assert pattern_for(row(['B', 'A'], 'B')) == 'return_to_earlier_hotel'
    assert pattern_for(row(['A'], 'C')) == 'new_hotel_after_exploration'
    assert pattern_for(row([], 'C')) == 'no_prior_hotel_reference'


def test_shuffle_preserves_multiset_padding_and_is_repeatable():
    h = torch.tensor([[2, 3, 4, 5, 0], [7, 7, 0, 0, 0], [0, 0, 0, 0, 0]])
    x = shuffled_histories(h, seed=5, offsets=range(3))
    assert torch.equal(x, shuffled_histories(h, seed=5, offsets=range(3)))
    for a, b in zip(h.tolist(), x.tolist()):
        assert sorted(a) == sorted(b)
        assert a.count(0) == b.count(0)
        assert b[-a.count(0):] == [0] * a.count(0) if a.count(0) else True


def test_expected_tie_neutral_metrics():
    scores = torch.tensor([[1., 1., 0.], [0., 2., 1.]])
    vals = expected_ranking_metrics(scores, torch.tensor([0, 1]))
    assert vals[0][0] == pytest.approx(.75)
    assert vals[1][0] == pytest.approx(1.)


def test_checkpoint_roundtrip_and_slice_population(tmp_path):
    path = tmp_path/'examples.jsonl'
    examples = [row(['A', 'B'], 'B'), row(['B', 'A'], 'B'),
                row(['A'], 'C'), row([], 'C')]
    path.write_text(''.join(json.dumps(e)+'\n' for e in examples))
    vocab = {'A': 2, 'B': 3, 'C': 4}
    model = create_model(5, 8)
    ckpt = tmp_path/'gru.pt'
    torch.save({'architecture':'hotel_id_gru', 'model':model.state_dict(),
                'vocabulary':vocab, 'dim':8, 'seed':42}, ckpt)
    report = run(ckpt, path, tmp_path/'report.json', split='validation', batch_size=2)
    assert report['original']['all']['n'] == 4
    assert report['shuffled']['all']['n'] == 4
    assert sum(report['original'][p]['n'] for p in (
        'immediate_continuation','return_to_earlier_hotel',
        'new_hotel_after_exploration','no_prior_hotel_reference')) == 4
    assert (tmp_path/'report.json').is_file()
    assert report['permutation']['eligible_length_ge_2'] == 2


def test_shuffle_deterministic_across_batches(tmp_path):
    path = tmp_path/'examples.jsonl'
    path.write_text(''.join(json.dumps(row(['A', 'B', 'C', 'A'], 'B'))+'\n'
                            for _ in range(5)))
    model = create_model(5, 8)
    vocab = {'A':2, 'B':3, 'C':4}
    a = analyze(model, path, vocab, split='validation', batch_size=2)
    b = analyze(model, path, vocab, split='validation', batch_size=4)
    assert a['original']['all']['mrr_neutral_ties'] == pytest.approx(b['original']['all']['mrr_neutral_ties'])
    assert a['shuffled']['all']['mrr_neutral_ties'] == pytest.approx(b['shuffled']['all']['mrr_neutral_ties'])
