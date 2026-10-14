"""Classification is mutually exclusive, causal, and insensitive to item ID spelling."""
import json

import pytest

from travelerreclab.data.audit_behavior_patterns import PATTERNS, audit, classify


def ex(history, target="B", time=200, **kwargs):
    return {"user_id": kwargs.get("user", "u"), "session_id": kwargs.get("session", "s"),
            "target_timestamp": time, "target_item": target,
            "candidate_items": ["A", "B", "C"], "history": [
                {"timestamp": 100 + i, "action_type": action, "reference": ref}
                for i, (action, ref) in enumerate(history)]}


def test_four_exclusive_patterns():
    cases = [
        ([("interaction item image", "A"), ("interaction item rating", "B")], "immediate_continuation"),
        ([("interaction item image", "B"), ("interaction item rating", "A")], "return_to_earlier_hotel"),
        ([("interaction item image", "A")], "new_hotel_after_exploration"),
        ([("search for destination", "Paris")], "no_prior_hotel_reference"),
    ]
    assert set(PATTERNS) == {classify(ex(history))["pattern"] for history, _ in cases}
    for history, expected in cases:
        assert classify(ex(history))["pattern"] == expected


def test_last_hotel_is_not_last_action():
    r = classify(ex([("interaction item image", "B"), ("search for destination", "Paris")]))
    assert r["pattern"] == "immediate_continuation"
    assert r["last_action"] == "search for destination"


def test_clickout_categories():
    first = classify(ex([("interaction item image", "A")]))
    repeat = classify(ex([("clickout item", "B"), ("interaction item image", "A")]))
    switch = classify(ex([("clickout item", "A")]))
    assert [a["clickout_pattern"] for a in (first, repeat, switch)] == [
        "first_clickout", "repeat_clickout", "switch_clickout"]


def test_position_and_time_buckets():
    e = ex([("interaction item image", "A")])
    r = classify(e)
    assert r["candidate_position"] == 2
    assert r["candidate_position_bucket"] == "2_5"
    assert r["last_event_gap_bucket"] == "61_300s"
    e["history"][0]["timestamp"] = 201
    with pytest.raises(ValueError, match="Future"):
        classify(e)


def test_aggregation_split_and_cross_tabs(tmp_path):
    path = tmp_path / "sample.jsonl"
    items = [ex([("interaction item image", "B")], time=1541030400),
             ex([("interaction item image", "A")], time=1541376000),
             ex([("search for destination", "Paris")], time=1541462400)]
    path.write_text("".join(json.dumps(e) + "\n" for e in items))
    report = audit(path)
    assert report["groups"]["all"]["n_examples"] == 3
    assert sum(x["n"] for x in report["groups"]["all"]["pattern"].values()) == 3
    assert [report["groups"][s]["n_examples"] for s in ("train", "validation", "test")] == [1, 1, 1]
    assert report["groups"]["all"]["cross"]["clickout_pattern"]["immediate_continuation"]["first_clickout"] == 1
    assert audit(path, max_examples=1)["groups"]["all"]["n_examples"] == 1
