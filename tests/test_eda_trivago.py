import pandas as pd

from travelerreclab.data.eda_trivago import audit, render_markdown


def test_streaming_eda_across_chunks(tmp_path):
    rows = [
        ["u1", "s1", 100, 1, "search for destination", "Paris", "FR", "Paris", "mobile", None, None, None],
        ["u1", "s1", 101, 2, "clickout item", "A", "FR", "Paris", "mobile", None, "A|B", "100|200"],
        ["u1", "s2", 102, 1, "clickout item", "C", "FR", "Paris", "mobile", None, "A|B", "100"],
        ["u2", "s1", 103, 1, "clickout item", "B", "FR", "Paris", "mobile", None, None, None],
    ]
    cols = ["user_id", "session_id", "timestamp", "step", "action_type", "reference", "platform", "city", "device", "current_filters", "impressions", "prices"]
    path = tmp_path / "train.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    result = audit(path, chunksize=2)
    assert result["rows"] == 4
    assert result["users"] == 2
    assert result["sessions"] == 3
    assert result["clickout_sessions"] == 3
    assert result["clickout_checks"]["total"] == 3
    assert result["clickout_checks"]["with_candidates"] == 2
    assert result["clickout_checks"]["target_in_candidates"] == 1
    assert result["clickout_checks"]["price_length_mismatch"] == 1
    assert result["events_per_session"]["max"] == 2
    assert "Click-out data quality" in render_markdown(result)


def test_missing_schema_errors(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"user_id": ["u1"]}).to_csv(path, index=False)
    import pytest
    with pytest.raises(ValueError, match="Missing required columns"):
        audit(path, chunksize=1)
