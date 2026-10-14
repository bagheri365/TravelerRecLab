import csv
import json
from datetime import datetime, timezone

import pytest

from travelerreclab.data.audit_hotel_metadata_v05 import summarize, history_hotels, behavior


def stamp(day):
    return int(datetime(2018, 11, day, tzinfo=timezone.utc).timestamp())


def example(day, target, candidates, history):
    return {"target_timestamp": stamp(day), "target_item": target,
            "candidate_items": candidates, "history": history}


def h(*items):
    return [{"action_type": "interaction item image", "reference": item} for item in items]


def fixture(tmp_path):
    examples = [
        example(4, "A", ["A", "B"], h("C")),
        example(5, "B", ["A", "B", "D"], h("A")),
        example(5, "C", ["A", "C"], h("B", "C", "A")),
        example(6, "A", ["A", "D"], h("D", "A")),
        example(6, "E", ["E", "D"], []),
    ]
    seq = tmp_path / "examples.jsonl"
    seq.write_text("".join(json.dumps(x) + "\n" for x in examples))
    meta = tmp_path / "metadata.csv"
    with meta.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "properties"])
        writer.writerows([["A", "WiFi|Pool"], ["B", "WiFi"], ["C", ""], ["D", "Parking"]])
    return seq, meta


def test_train_membership_isolation_and_metadata_rates(tmp_path):
    seq, meta = fixture(tmp_path)
    report = summarize(seq, meta)
    assert report["train_vocabulary_unique_ids"] == 3  # A, B, C (history)
    assert report["train_target_unique_ids"] == 1
    val = report["by_split_and_behavior"]["validation"]["all"]
    assert val["examples"] == 2
    assert val["rates"]["target_in_train_vocab_rate"] == 1
    assert val["rates"]["target_in_train_targets_rate"] == 0
    assert val["rates"]["target_with_properties_rate"] == .5
    assert val["rates"]["candidate_slots_in_metadata_rate"] == 1
    test = report["by_split_and_behavior"]["test"]["all"]
    assert test["rates"]["target_in_train_vocab_rate"] == .5
    assert test["rates"]["target_in_metadata_rate"] == .5


def test_behavior_segments_follow_retained_hotel_history(tmp_path):
    seq, meta = fixture(tmp_path)
    report = summarize(seq, meta)["by_split_and_behavior"]
    assert report["validation"]["return_to_earlier_hotel"]["examples"] == 1
    assert report["validation"]["new_hotel_after_exploration"]["examples"] == 1
    assert report["test"]["immediate_continuation"]["examples"] == 1
    assert report["test"]["no_prior_hotel_reference"]["examples"] == 1
    ex = example(6, "A", ["A"], [{"action_type":"search for destination", "reference":"A"}])
    assert history_hotels(ex) == []
    assert behavior(ex, []) == "no_prior_hotel_reference"


def test_missing_metadata_and_empty_properties_are_distinct(tmp_path):
    seq, meta = fixture(tmp_path)
    rows = summarize(seq, meta)
    assert rows["metadata"]["relevant_ids_with_metadata"] == 4
    assert rows["metadata"]["relevant_ids_with_nonempty_properties"] == 3
    assert rows["metadata"]["property_tag_vocabulary_size_relevant"] == 3


def test_invalid_metadata_schema(tmp_path):
    seq, _ = fixture(tmp_path)
    meta = tmp_path / "wrong.csv"
    meta.write_text("hotel_id,tags\nA,Wifi\n")
    with pytest.raises(ValueError, match="item_id and properties"):
        summarize(seq, meta)
