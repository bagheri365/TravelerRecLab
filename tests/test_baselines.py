import json
import math

import pytest

from travelerreclab.evaluation.baselines import (
    Metrics, evaluate, fit_popularity, rank_of_target, score_candidates, split_of,
)


def example(timestamp, target="A", history=None, candidates=None):
    return {"user_id": "u", "session_id": "s", "target_timestamp": timestamp,
            "target_step": 3, "target_item": target,
            "candidate_items": candidates or ["B", "A", "C"], "history": history or []}


def test_time_split_is_utc_and_boundary_exact():
    assert split_of(1541376000) == "validation"  # 2018-11-05 UTC
    assert split_of(1541462400) == "test"
    assert split_of(1541289600) == "train"


def test_train_only_popularity(tmp_path):
    p = tmp_path / "examples.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        example(1541289600, "A"), example(1541376000, "B"), example(1541462400, "C")]) + "\n")
    assert fit_popularity(p, val_day="2018-11-05", test_day="2018-11-06") == {"A": 1}
    result = evaluate(p)
    assert result["metrics"]["validation"]["popularity"]["n"] == 1
    assert result["metrics"]["test"]["popularity"]["n"] == 1


def test_last_item_ignores_destination_and_ties():
    e = example(100, history=[
        {"action_type": "interaction item image", "reference": "A", "timestamp": 40},
        {"action_type": "search for destination", "reference": "Paris", "timestamp": 60},
    ])
    assert score_candidates(e, "last_item", {}) == [0, 1, 0]
    assert rank_of_target(e, [0, 1, 0]) == 1
    assert rank_of_target(e, [0, 0, 0]) == 2  # candidate order B, A, C


def test_recency_prefers_recent_and_action_weights():
    e = example(100, history=[
        {"action_type": "interaction item image", "reference": "A", "timestamp": 0},
        {"action_type": "clickout item", "reference": "B", "timestamp": 99},
    ])
    scores = score_candidates(e, "recency_weighted", {}, decay_seconds=10)
    assert scores[0] > scores[1]
    assert score_candidates(e, "action_weighted", {}) == [2, 1, 0]


def test_random_is_reproducible_and_metrics():
    e = example(100)
    assert score_candidates(e, "random", {}) == score_candidates(e, "random", {})
    m = Metrics()
    m.add(1)
    m.add(2)
    r = m.result()
    assert r["mrr"] == .75
    assert r["recall@5"] == 1
    assert r["ndcg@5"] == pytest.approx((1 + 1 / math.log2(3)) / 2)


def test_invalid_and_empty_examples(tmp_path):
    p = tmp_path / "examples.jsonl"
    p.write_text(json.dumps(example(1541462400, target="Z")) + "\n")
    r = evaluate(p)
    assert r["counts"] == {"invalid_candidates": 1}
    assert r["metrics"]["test"]["random"]["mrr"] is None
    with pytest.raises(ValueError):
        evaluate(p, max_examples=0)
