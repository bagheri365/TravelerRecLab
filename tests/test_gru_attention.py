"""Candidate-aware attention GRU: shape, leakage/masking, and train controls."""
import json
import pytest

from travelerreclab.models.gru_attention import create_model, train
from travelerreclab.models.mean_pool import make_batch


def row(day, history=None, target='B'):
    from datetime import datetime, timezone
    return {'target_timestamp': int(datetime(2018, 11, day, tzinfo=timezone.utc).timestamp()),
            'target_item': target, 'candidate_items': ['A', 'B', 'C'],
            'history': [{'action_type': 'interaction item image', 'reference': item}
                        for item in (history if history is not None else ['A', 'B'])]}


def test_variable_length_empty_and_candidate_padding():
    torch = pytest.importorskip('torch')
    h, c, _ = make_batch([row(4, ['A', 'B']), row(4, []), row(4, ['A'])],
                         {'A': 2, 'B': 3, 'C': 4})
    c[0, -1] = 0
    scores = create_model(5, 8)(h, c)
    assert scores.shape == (3, 3)
    assert scores[0, -1].item() == float('-inf')
    assert torch.isfinite(scores[1]).all()


def test_padding_does_not_change_scores():
    torch = pytest.importorskip('torch')
    model = create_model(5, 8)
    c = torch.tensor([[2, 3, 4]])
    assert torch.allclose(model(torch.tensor([[2, 3]]), c),
                          model(torch.tensor([[2, 3, 0, 0]]), c), atol=1e-6)


def test_candidate_queries_produce_candidate_specific_context():
    torch = pytest.importorskip('torch')
    torch.manual_seed(7)
    model = create_model(5, 8)
    h = torch.tensor([[2, 3, 4]])
    c = torch.tensor([[2, 3, 4]])
    out = model(h, c)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert model.query.weight.grad is not None
    assert model.query.weight.grad.abs().sum().item() > 0
    assert model.mix_logit.grad is not None


def test_training_is_reproducible_and_checkpoint_loadable(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'small.jsonl'
    rows = [row(4, ['A', 'B']), row(4, ['B', 'A'], 'A'),
            row(5, ['A', 'B']), row(6, ['A', 'B'])]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    a = train(path, tmp_path / 'a.json', epochs=2, batch_size=2, dim=8)
    b = train(path, tmp_path / 'b.json', epochs=2, batch_size=2, dim=8)
    assert a['training'] == b['training']
    assert a['test'] == b['test']
    assert a['best_epoch'] in (1, 2)
    checkpoint = torch.load(tmp_path / 'a.pt', map_location='cpu', weights_only=True)
    model = create_model(max(checkpoint['vocabulary'].values()) + 1, checkpoint['dim'])
    model.load_state_dict(checkpoint['model'])
    assert checkpoint['architecture'] == 'hotel_id_candidate_attention_gru'


def test_invalid_hyperparameters(tmp_path):
    pytest.importorskip('torch')
    with pytest.raises(ValueError, match='Invalid'):
        train(tmp_path / 'missing.jsonl', tmp_path / 'out.json', dim=0)
