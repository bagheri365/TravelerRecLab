"""Raw-level semantics and truncation checks on fully observed toy sessions."""
import pandas as pd

from travelerreclab.data.audit_raw_semantics import audit_frames


def row(step, action, ref=None, candidates="A|B", prices="1|2", session="s"):
    return {"user_id": "u", "session_id": session, "timestamp": 100 + step,
            "step": step, "action_type": action, "reference": ref,
            "impressions": candidates if action == "clickout item" else None,
            "prices": prices if action == "clickout item" else None}


def test_clickout_full_history_partition_and_builder_eligibility():
    rows = [row(1, "clickout item", "A"), row(2, "interaction item image", "B"),
            row(3, "clickout item", "B"), row(4, "clickout item", "B"),
            row(5, "clickout item", "A")]
    result = audit_frames(pd.DataFrame(rows), {"A", "B"}, max_history=2)
    raw = result["raw"]
    assert raw["raw_clickouts"] == 4
    assert (raw["first_clickout"], raw["switch_clickout"], raw["repeat_clickout"]) == (1, 2, 1)
    assert raw["zero_history"] == 1
    assert raw["eligible_min_history_1"] == 3
    assert raw["eligible_genuinely_truncated"] == 2
    assert result["checks"]["clickout_partition_matches"]


def test_exact_max_history_not_truncated():
    rows = [row(1, "interaction item image", "A"), row(2, "interaction item info", "B"),
            row(3, "clickout item", "B")]
    raw = audit_frames(pd.DataFrame(rows), {"A", "B"}, max_history=2)["raw"]
    assert raw["exactly_max_history_before_target"] == 1
    assert raw.get("truncated_if_emitted", 0) == 0


def test_target_was_seen_but_lost_in_retained_window():
    rows = [row(1, "interaction item image", "B"), row(2, "search for destination", "Paris"),
            row(3, "interaction item image", "A"), row(4, "clickout item", "B")]
    raw = audit_frames(pd.DataFrame(rows), {"A", "B"}, max_history=2)["raw"]
    assert raw["eligible_target_lost_by_truncation"] == 1
    assert raw["eligible_target_seen_full_history"] == 1


def test_search_for_item_metadata_overlap_is_not_semantics_proof():
    rows = [row(1, "search for item", "A"), row(2, "search for item", "not_a_hotel"),
            row(3, "search for item", None), row(4, "clickout item", "B")]
    r = audit_frames(pd.DataFrame(rows), {"A", "B"})
    assert r["search_for_item"] == {"in_metadata": 1, "missing": 1, "nonempty": 2, "total": 3}


def test_session_sort_is_chronological_not_input_order():
    rows = [row(3, "clickout item", "B"), row(1, "interaction item image", "A"),
            row(2, "clickout item", "A")]
    raw = audit_frames(pd.DataFrame(rows), {"A", "B"})["raw"]
    assert raw["first_clickout"] == 1
    assert raw["switch_clickout"] == 1
    assert raw["eligible_min_history_1"] == 2
