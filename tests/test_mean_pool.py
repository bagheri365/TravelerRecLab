import json
import pytest

from travelerreclab.models.mean_pool import build_vocabulary, create_model, encode, evaluate, hotel_history, make_batch, train


def ex(day, target='B', candidates=None, history=None):
    from datetime import datetime, timezone
    return {'target_timestamp': int(datetime(2018, 11, day, tzinfo=timezone.utc).timestamp()),
            'target_item': target, 'candidate_items': candidates or ['A', 'B', 'C'],
            'history': history or [{'action_type': 'interaction item image', 'reference': 'A'}]}


def fixture(tmp_path):
    p = tmp_path / 'examples.jsonl'
    entries = [ex(4, 'B'), ex(4, 'A'), ex(5, 'C'), ex(6, 'C')]
    p.write_text(''.join(json.dumps(e) + '\n' for e in entries))
    return p


def test_hotel_history_ignores_destinations():
    e = ex(4, history=[{'action_type': 'search for destination', 'reference': 'Paris'},
                       {'action_type': 'clickout item', 'reference': 'A'}])
    assert hotel_history(e) == ['A']


def test_vocab_is_train_only(tmp_path):
    p = tmp_path / 'x.jsonl'
    p.write_text('\n'.join(json.dumps(x) for x in [ex(4, 'B', ['A', 'B']),
                                ex(5, 'X', ['X', 'Y']), ex(6, 'Z', ['Z', 'W'])]) + '\n')
    v = build_vocabulary(p)
    assert set(v) == {'A', 'B'}
    assert encode(ex(5, 'X', ['X', 'Y']), v)[1] == [1, 1]


def test_batch_padding_and_mask():
    torch = pytest.importorskip('torch')
    vocab = {'A': 2, 'B': 3, 'C': 4}
    rows = [ex(4), ex(4, history=[{'action_type': 'search for destination', 'reference': 'Paris'}])]
    h, c, y = make_batch(rows, vocab)
    assert h.tolist() == [[2], [0]]
    assert y.tolist() == [1, 1]
    m = create_model(5, 8)
    assert torch.isfinite(m(h, c)).all()


def test_train_smoke_deterministic_and_no_test_selection(tmp_path):
    pytest.importorskip('torch')
    p = fixture(tmp_path)
    a = train(p, tmp_path / 'a.json', epochs=2, dim=8, batch_size=2)
    b = train(p, tmp_path / 'b.json', epochs=2, dim=8, batch_size=2)
    assert a['test'] == b['test']
    assert a['best_epoch'] == b['best_epoch']
    assert a['training'][0]['validation']['n'] == 1
    assert a['test']['n'] == 1
    assert (tmp_path / 'a.pt').exists()


def test_tie_neutral_unknown_candidates(tmp_path):
    pytest.importorskip('torch')
    p = fixture(tmp_path)
    vocab = {'A': 2, 'B': 3}
    model = create_model(4, 8)
    result = evaluate(model, p, 'test', vocab, 2)
    assert result['n'] == 1
    assert result['unknown_target_rate'] == 1
    assert 0 < result['mrr_neutral_ties'] <= 1
