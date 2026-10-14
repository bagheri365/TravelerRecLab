import json
import math

import pytest

from travelerreclab.evaluation.hybrid_position import behavioral_pattern, compute, hybrid_scores


def example(day, target="B", refs=("A",), candidates=("A", "B", "C")):
    from datetime import datetime
    ts = int(datetime.fromisoformat(f"{day}T12:00:00+00:00").timestamp())
    return {"target_timestamp": ts, "target_item": target, "candidate_items": list(candidates),
            "history": [{"timestamp": ts - 10, "action_type": "interaction item image", "reference": r}
                        for r in refs]}


def test_pattern_partition_and_nonhotel_reference():
    assert behavioral_pattern(example("2018-11-05", "B", ("A", "B"))) == "immediate_continuation"
    assert behavioral_pattern(example("2018-11-05", "A", ("A", "B"))) == "return_to_earlier_hotel"
    assert behavioral_pattern(example("2018-11-05", "C", ("A", "B"))) == "new_hotel_after_exploration"
    assert behavioral_pattern(example("2018-11-05", "B", ())) == "no_prior_hotel_reference"
    e = example("2018-11-05", "B", ())
    e["history"] = [{"action_type": "search for destination", "reference": "B"}]
    assert behavioral_pattern(e) == "no_prior_hotel_reference"


def test_position_scale_and_boost():
    e = example("2018-11-05", "B", ("B",))
    assert hybrid_scores(e, 0) == [1., .5, 0.]
    assert hybrid_scores(e, 1) == [1., 1.5, 0.]
    assert hybrid_scores(example("2018-11-05", "A", (), ("A",)), 2) == [1.]
    with pytest.raises(ValueError):
        hybrid_scores(e, -1)


def test_nonhotel_actions_do_not_boost():
    e = example("2018-11-05", "B", ())
    e["history"] = [{"action_type": "filter selection", "reference": "B"}]
    assert hybrid_scores(e, 2) == hybrid_scores(e, 0)


def test_selected_boost_depends_on_validation_not_test(tmp_path):
    rows = [example("2018-11-05", "A", ("A",)),
            example("2018-11-05", "B", ("B",)),
            example("2018-11-06", "C", ("A",))]
    f = tmp_path / "rows.jsonl"
    f.write_text("".join(json.dumps(e) + "\n" for e in rows))
    first = compute(f, boosts=(0, 0.5, 2))
    rows[-1]["target_item"] = "A"
    f.write_text("".join(json.dumps(e) + "\n" for e in rows))
    second = compute(f, boosts=(0, 0.5, 2))
    assert first["selection"] == second["selection"]
    assert first["metrics"]["validation"] == second["metrics"]["validation"]


def test_position_only_and_segments(tmp_path):
    rows = [example("2018-11-05", "B", ("A",)),
            example("2018-11-06", "B", ("B",))]
    f = tmp_path / "rows.jsonl"
    f.write_text("".join(json.dumps(e) + "\n" for e in rows))
    r = compute(f, boosts=(0., 2.))
    assert r["metrics"]["validation"]["boost_0"]["all"]["mrr_neutral_ties"] == .5
    assert r["metrics"]["validation"]["boost_0"]["new_hotel_after_exploration"]["n"] == 1
    assert r["metrics"]["test"]["boost_2"]["immediate_continuation"]["n"] == 1


def test_empty_validation_and_bad_grid(tmp_path):
    f = tmp_path / "rows.jsonl"
    f.write_text(json.dumps(example("2018-11-06")) + "\n")
    with pytest.raises(ValueError, match="No validation"):
        compute(f)
    with pytest.raises(ValueError, match="unique nonnegative"):
        compute(f, boosts=(0, 0))
