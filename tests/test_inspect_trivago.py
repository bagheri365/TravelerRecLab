import pandas as pd
import pytest

from travelerreclab.data.inspect_trivago import summarize_events


def test_summary_uses_composite_session_key(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"user_id": "u1", "session_id": "s1", "timestamp": 1, "step": 1,
             "action_type": "search for destination", "reference": "Paris",
             "impressions": None, "prices": None},
            {"user_id": "u1", "session_id": "s1", "timestamp": 2, "step": 2,
             "action_type": "clickout item", "reference": "101",
             "impressions": "101|102", "prices": "100|110"},
            {"user_id": "u2", "session_id": "s1", "timestamp": 3, "step": 1,
             "action_type": "clickout item", "reference": "102",
             "impressions": "102|103", "prices": "110|130"},
        ]
    ).to_csv(path, index=False)
    result = summarize_events(path)
    assert result["rows"] == 3
    assert result["users"] == 2
    assert result["sessions"] == 2
    assert result["actions"]["clickout item"] == 2
    assert result["events_per_session"]["max"] == 2


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_events(tmp_path / "unknown.csv")


def test_missing_required_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"user_id": ["u1"], "action_type": ["clickout item"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        summarize_events(path)
