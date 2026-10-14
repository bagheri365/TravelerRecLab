import pandas as pd
import pytest

from travelerreclab.data.audit_final import audit_first_events, audit_item_coverage, run


COLUMNS = ["user_id", "session_id", "timestamp", "step", "action_type", "reference", "impressions", "prices"]


def fixtures(tmp_path):
    events = tmp_path / "events.csv"
    pd.DataFrame([
        ("u", "s", 20, 2, "clickout item", "B", "A|B", "1|2"),
        ("u", "s", 10, 1, "interaction item image", "A", None, None),
        ("u", "s", 30, 3, "clickout item", "A", "A|B", "1|2"),
        ("v", "t", 1, 1, "clickout item", "Z", "Z", "5"),
        ("v", "t", 2, 2, "interaction item image", "Z", None, None),
    ], columns=COLUMNS).to_csv(events, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"item_id": ["A", "B"], "properties": ["Pool|WiFi", "WiFi"]}).to_csv(meta, index=False)
    return events, meta


def test_reconcile_first_event(tmp_path):
    events, _ = fixtures(tmp_path)
    x = audit_first_events(events, chunksize=2)
    assert x["predicted_sorted_minus_source_eligible"] == 1
    assert x["sessions_with_different_first_event"] == 1
    assert x["source_first_valid_clickouts"] == 2
    assert x["sorted_first_valid_clickouts"] == 1


def test_coverage_and_properties(tmp_path):
    events, meta = fixtures(tmp_path)
    x = audit_item_coverage(events, meta, chunksize=2)
    assert x["distinct_clicked_items"] == 3
    assert x["clicked_items_with_metadata"] == 2
    assert x["clickout_weighted_metadata_coverage"] == pytest.approx(2 / 3)
    assert ("WiFi", 2) in x["top_metadata_properties"]


def test_report(tmp_path):
    events, meta = fixtures(tmp_path)
    result = run(events, meta, tmp_path / "out", chunksize=2, expected_delta=1)
    assert result["difference_matches_reported"]
    assert (tmp_path / "out" / "report.md").exists()
    assert (tmp_path / "out" / "summary.json").exists()


def test_missing_metadata_schema(tmp_path):
    events, meta = fixtures(tmp_path)
    pd.DataFrame({"wrong": ["A"]}).to_csv(meta, index=False)
    with pytest.raises(ValueError, match="Metadata must have"):
        audit_item_coverage(events, meta)
