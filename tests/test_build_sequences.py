import json

import pandas as pd
import pytest

from travelerreclab.data.build_sequences import build_examples, build_file


def event(user, session, time, step, action, ref, impressions=None, prices=None):
    return {
        "user_id": user, "session_id": session, "timestamp": time, "step": step,
        "action_type": action, "reference": ref, "impressions": impressions, "prices": prices,
    }


def test_sort_causal_history_and_independent_sessions():
    rows = [
        event("u1", "s1", 12, 3, "clickout item", "b", "a|b", "100|120"),
        event("u2", "s1", 10, 1, "search for destination", "Rome"),
        event("u1", "s1", 10, 1, "search for destination", "Paris"),
        event("u1", "s1", 11, 2, "interaction item info", "a"),
        event("u2", "s1", 11, 2, "clickout item", "c", "c|d", "80|90"),
    ]
    result = build_examples(pd.DataFrame(rows))
    assert len(result) == 2
    paris = next(x for x in result if x["user_id"] == "u1")
    assert [e["reference"] for e in paris["history"]] == ["Paris", "a"]
    assert paris["target_item"] == "b"
    assert paris["candidate_items"] == ["a", "b"]
    assert all(e["timestamp"] < paris["target_timestamp"] for e in paris["history"])
    rome = next(x for x in result if x["user_id"] == "u2")
    assert [e["reference"] for e in rome["history"]] == ["Rome"]


def test_same_timestamp_uses_step_and_truncates_history():
    rows = [
        event("u", "s", 10, 3, "clickout item", "h", "h|j", "1|2"),
        event("u", "s", 10, 1, "search for destination", "Paris"),
        event("u", "s", 10, 2, "interaction item info", "j"),
    ]
    examples = build_examples(pd.DataFrame(rows), max_history=1)
    assert [x["step"] for x in examples[0]["history"]] == [2]


def test_bad_candidates_and_missing_labels_are_skipped():
    rows = [
        event("u", "s", 1, 1, "search for destination", "Paris"),
        event("u", "s", 2, 2, "clickout item", "x", "a|b", "1|2"),
        event("u", "s", 3, 3, "clickout item", "a", "a|b", "1"),
        event("u", "s", 4, 4, "clickout item", "b", "a|b", "1|2"),
    ]
    assert [x["target_item"] for x in build_examples(pd.DataFrame(rows))] == ["b"]


def test_validation():
    frame = pd.DataFrame([event("u", "s", 1, 1, "clickout item", "a")])
    with pytest.raises(ValueError, match="max_history"):
        build_examples(frame, max_history=0)
    with pytest.raises(ValueError, match="must not be null"):
        build_examples(frame.assign(timestamp=None))


def test_writes_jsonl(tmp_path):
    src = tmp_path / "events.csv"
    dst = tmp_path / "sub" / "examples.jsonl"
    pd.DataFrame([
        event("u", "s", 1, 1, "search for destination", "Paris"),
        event("u", "s", 2, 2, "clickout item", "a", "a|b", "1|2"),
    ]).to_csv(src, index=False)
    assert build_file(src, dst) == 1
    result = json.loads(dst.read_text().strip())
    assert result["target_item"] == "a"
    assert len(result["history"]) == 1
