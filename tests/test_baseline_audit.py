import json
import math
from travelerreclab.evaluation.baseline_audit import RankingMetrics, compute, segments


def event(day, target, history, candidates=("A", "B", "C")):
    from datetime import datetime, timezone
    ts = int(datetime.fromisoformat(day + "T12:00:00+00:00").timestamp())
    return {"target_timestamp": ts, "target_item": target, "candidate_items": list(candidates),
            "history": history, "user_id": "U", "session_id": day, "target_step": 2}


def hist(ref="A", n=1):
    return [{"action_type": "interaction item image", "reference": ref, "timestamp": 1541030000}] * n


def test_tied_scores_neutral_metric():
    m = RankingMetrics()
    m.add([0, 0, 0], 2)
    got = m.result()
    assert math.isclose(got["mrr_logged_ties"], 1/3)
    assert math.isclose(got["mrr_neutral_ties"], (1+1/2+1/3)/3)


def test_no_ties_same_rank():
    m = RankingMetrics()
    m.add([0, 3, 1], 1)
    assert m.result()["mrr_logged_ties"] == m.result()["mrr_neutral_ties"] == 1


def test_slice_semantics():
    e = event("2018-11-05", "A", hist("A", 6))
    assert segments(e) == ("all", "target_seen", "history_6_20")
    e["history"] = [{"action_type": "search for destination", "reference": "A"}]
    assert "target_unseen" in segments(e)


def test_selection_uses_validation_only(tmp_path):
    rows = [event("2018-11-01", "A", hist("A")),
            event("2018-11-05", "A", hist("A")),
            event("2018-11-06", "B", hist("B"))]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in rows))
    one = compute(p, decay_candidates=(10., 3600.))
    rows[-1]["target_item"] = "C"
    p.write_text("".join(json.dumps(x) + "\n" for x in rows))
    two = compute(p, decay_candidates=(10., 3600.))
    assert one["selection"] == two["selection"]
    assert one["metrics"]["validation"] == two["metrics"]["validation"]


def test_position_baseline(tmp_path):
    rows = [event("2018-11-01", "A", hist()), event("2018-11-05", "B", hist())]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in rows))
    result = compute(p, decay_candidates=(100.,))
    m = result["metrics"]["validation"]["position"]["all"]
    assert m["mrr_logged_ties"] == m["mrr_neutral_ties"] == .5
