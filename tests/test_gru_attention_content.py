import csv

import pytest
import torch

from travelerreclab.models.gru_attention_content import (
    create_model, load_properties, make_content_batch,
)


def example(history, candidates=('10', '20'), target='10'):
    return {'history': [{'action_type': 'interaction item info', 'reference': x} for x in history],
            'candidate_items': list(candidates), 'target_item': target}


def test_metadata_parser(tmp_path):
    path = tmp_path / 'metadata.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'properties'])
        writer.writerow(['10', 'WiFi|Hotel|WiFi'])
        writer.writerow(['20', 'Pool'])
    props, tags = load_properties(path)
    assert len(tags) == 3
    assert len(props['10']) == 2
    assert props['10'] != props['20']


def test_batch_preserves_raw_unknown_properties():
    props = {'10': (1,), '20': (2,), '30': (3,)}
    h, c, ht, ct, target = make_content_batch(
        [example(['10'], ('20', '30'), '30')], {'10': 2}, props)
    assert c.tolist() == [[1, 1]]  # both ID unknown
    assert ct[0, 0, 0] != ct[0, 1, 0]  # but properties distinguish candidates
    assert target.tolist() == [1]


def test_padding_and_empty_history():
    model = create_model(6, 4, 8)
    rows = [example(['10'], ('10', '20')), example([], ('10', '20'))]
    h, c, ht, ct, _ = make_content_batch(rows, {'10': 2, '20': 3}, {'10': (1,2), '20': (3,)})
    scores = model(h, c, ht, ct)
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(model(h, c, ht, ct)).all()


def test_unseen_candidate_content_changes_score():
    torch.manual_seed(7)
    model = create_model(3, 3, 8)
    h = torch.zeros((1, 1), dtype=torch.long)
    c = torch.ones((1, 2), dtype=torch.long)
    ht = torch.zeros((1, 1, 1), dtype=torch.long)
    ct = torch.tensor([[[1], [2]]])
    scores = model(h, c, ht, ct)
    assert scores[0, 0] != scores[0, 1]
    scores.sum().backward()
    assert model.tags.weight.grad[1].abs().sum() > 0


def test_invalid_weight_and_missing_tags():
    with pytest.raises(ValueError):
        create_model(3, 3, 8, 1.5)
    model = create_model(3, 3, 8)
    h = torch.zeros((1, 1), dtype=torch.long)
    c = torch.tensor([[1, 1]])
    tags = torch.zeros((1, 1, 1), dtype=torch.long)
    assert torch.isfinite(model(h, c, tags, torch.zeros((1, 2, 1), dtype=torch.long))).all()
