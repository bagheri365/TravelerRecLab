import pandas as pd
import pytest

from travelerreclab.data.validate_raw import audit_events, audit_metadata, run, split_list


def test_split_list():
    assert split_list("a| b") == ["a", "b"]
    assert split_list(pd.NA) == []


def test_exclusions_and_order(tmp_path):
    events = pd.DataFrame([
        ("u", "s", 10, 1, "clickout item", "A", "A|B", "1|2"),
        ("u", "s", 11, 2, "interaction item image", "A", None, None),
        ("u", "s", 10, 2, "clickout item", "B", "A|B", "1|2"),
        ("u", "s", 12, 2, "clickout item", "C", "A|B", "1|2"),
        ("u", "s", 13, 3, "clickout item", "B", "A|B", "1|2"),
    ], columns=["user_id", "session_id", "timestamp", "step", "action_type", "reference", "impressions", "prices"])
    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)
    counts = audit_events(path, chunksize=2)["counts"]
    assert counts["clickouts"] == 4
    assert counts["excluded_short_history"] == 1
    assert counts["excluded_target_not_in_candidates"] == 1
    assert counts["eligible_in_source_order"] == 2
    assert counts["timestamp_reversals"] == 1
    assert counts["duplicate_step_rows"] == 2


def test_metadata_and_reports(tmp_path):
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"item_id": ["A", "B"], "properties": ["Pool", None]}).to_csv(meta, index=False)
    assert audit_metadata(meta, ["A", "Z"])["clicked_item_metadata_coverage"] == .5
    events = tmp_path / "events.csv"
    pd.DataFrame([("u", "s", 10, 1, "clickout item", "A", "A", "1")],
                 columns=["user_id", "session_id", "timestamp", "step", "action_type", "reference", "impressions", "prices"]).to_csv(events, index=False)
    result = run(events, meta, tmp_path / "out", chunksize=1, emitted_examples=0)
    assert result["raw"]["counts"]["excluded_short_history"] == 1
    assert (tmp_path / "out" / "report.md").exists()


def test_invalid_metadata(tmp_path):
    p = tmp_path / "metadata.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="item_id"):
        audit_metadata(p)
