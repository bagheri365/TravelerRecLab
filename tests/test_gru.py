"""Small synthetic checks for the GRU history encoder and training contract."""
import json

import pytest

from travelerreclab.models.gru import create_model, train
from travelerreclab.models.mean_pool import make_batch


def row(day, history=None, target='B'):
    from datetime import datetime, timezone
    return {'target_timestamp': int(datetime(2018, 11, day, tzinfo=timezone.utc).timestamp()),
            'target_item': target, 'candidate_items': ['A', 'B', 'C'],
            'history': [{'action_type': 'interaction item image', 'reference': item}
                        for item in (history if history is not None else ['A', 'B'])]}


def test_gru_output_mask_and_empty_history():
    torch = pytest.importorskip('torch')
    vocab = {'A': 2, 'B': 3, 'C': 4}
    examples = [row(4, ['A', 'B']), row(4, []), row(4, ['A'])]
    h, c, _ = make_batch(examples, vocab)
    c[0, -1] = 0
    model = create_model(5, 8)
    scores = model(h, c)
    assert scores.shape == (3, 3)
    assert scores[0, -1] == float('-inf')
    assert torch.isfinite(scores[1]).all()


def test_order_affects_gru_but_not_mean_pool():
    torch = pytest.importorskip('torch')
    from travelerreclab.models.mean_pool import create_model as mean_model
    torch.manual_seed(3)
    v = {'A': 2, 'B': 3, 'C': 4}
    h, c, _ = make_batch([row(4, ['A', 'B']), row(4, ['B', 'A'])], v)
    gru_scores = create_model(5, 8)(h, c)
    pooled_scores = mean_model(5, 8)(h, c)
    assert torch.allclose(pooled_scores[0], pooled_scores[1])
    assert not torch.allclose(gru_scores[0], gru_scores[1])


def test_padded_history_does_not_change_valid_sequence():
    torch = pytest.importorskip('torch')
    model = create_model(5, 8)
    cand = torch.tensor([[2, 3, 4]])
    short = torch.tensor([[2, 3]])
    padded = torch.tensor([[2, 3, 0, 0]])
    assert torch.allclose(model(short, cand), model(padded, cand), atol=1e-6)


def test_train_smoke_reproducible_and_checkpoint_selected(tmp_path):
    pytest.importorskip('torch')
    path = tmp_path / 'ex.jsonl'
    entries = [row(4, ['A', 'B']), row(4, ['B', 'A'], 'A'),
               row(5, ['A', 'B']), row(6, ['A', 'B'])]
    path.write_text(''.join(json.dumps(e) + '\n' for e in entries))
    first = train(path, tmp_path / 'a.json', epochs=2, batch_size=2, dim=8)
    second = train(path, tmp_path / 'b.json', epochs=2, batch_size=2, dim=8)
    assert first['training'] == second['training']
    assert first['test'] == second['test']
    assert first['test']['n'] == 1
    assert first['best_epoch'] in (1, 2)
    assert (tmp_path / 'a.pt').is_file()


def test_invalid_hyperparameters(tmp_path):
    pytest.importorskip('torch')
    with pytest.raises(ValueError, match='Invalid'):
        train(tmp_path / 'missing.jsonl', tmp_path / 'out.json', dim=0)
