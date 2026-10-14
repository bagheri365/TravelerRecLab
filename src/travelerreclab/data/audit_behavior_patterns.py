"""Read-only taxonomy of existing Trivago next-click-out JSONL examples.

Unit of analysis: emitted prediction example, not independent traveler or session.
Historical hotel references follow the same provisional action contract as v0.2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from travelerreclab.evaluation.baselines import HOTEL_ACTIONS, read_examples, split_of

PATTERNS = (
    "immediate_continuation", "return_to_earlier_hotel",
    "new_hotel_after_exploration", "no_prior_hotel_reference",
)
CLICKOUT_PATTERNS = ("first_clickout", "repeat_clickout", "switch_clickout")
HISTORY_BUCKETS = ("1_5", "6_20", "21_49", "50_truncated")
POSITION_BUCKETS = ("1", "2_5", "6_10", "11_plus")
GAP_BUCKETS = ("0s", "1_10s", "11_60s", "61_300s", "301_3600s", "over_3600s")


def _bucket(value: int, *, kind: str) -> str:
    if kind == "history":
        return "1_5" if value <= 5 else "6_20" if value <= 20 else "21_49" if value < 50 else "50_truncated"
    if kind == "position":
        return "1" if value == 1 else "2_5" if value <= 5 else "6_10" if value <= 10 else "11_plus"
    if kind == "gap":
        return ("0s" if value <= 0 else "1_10s" if value <= 10 else "11_60s" if value <= 60
                else "61_300s" if value <= 300 else "301_3600s" if value <= 3600 else "over_3600s")
    raise ValueError(kind)


def classify(example: dict) -> dict:
    """Pure classification. All rules use only the retained pre-target history.

    search-for-item is provisionally hotel-typed, matching v0.2 baseline code.
    The term 'immediate' means last *hotel reference*, not necessarily last event.
    """
    history = example["history"]
    target = str(example["target_item"])
    hotels = [str(e["reference"]) for e in history
              if e.get("action_type") in HOTEL_ACTIONS and e.get("reference")]
    if not hotels:
        pattern = "no_prior_hotel_reference"
    elif hotels[-1] == target:
        pattern = "immediate_continuation"
    elif target in hotels:
        pattern = "return_to_earlier_hotel"
    else:
        pattern = "new_hotel_after_exploration"
    clickouts = [str(e["reference"]) for e in history
                 if e.get("action_type") == "clickout item" and e.get("reference")]
    clickout_pattern = ("first_clickout" if not clickouts else
                        "repeat_clickout" if clickouts[-1] == target else "switch_clickout")
    candidates = [str(x) for x in example["candidate_items"]]
    if target not in candidates:
        raise ValueError("Target not present in candidates")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate candidates")
    if not history:
        raise ValueError("Expected >=1 historical event in emitted benchmark")
    now = int(example["target_timestamp"])
    last_event_gap = now - int(history[-1]["timestamp"])
    if last_event_gap < 0:
        raise ValueError("Future event in history")
    target_last = next((int(e["timestamp"]) for e in reversed(history)
                       if e.get("action_type") in HOTEL_ACTIONS and str(e.get("reference")) == target), None)
    last_hotel_time = next((int(e["timestamp"]) for e in reversed(history)
                            if e.get("action_type") in HOTEL_ACTIONS and e.get("reference")), None)
    for stamp in (target_last, last_hotel_time):
        if stamp is not None and stamp > now:
            raise ValueError("Future hotel event in history")
    return {
        "pattern": pattern,
        "clickout_pattern": clickout_pattern,
        "last_action": history[-1].get("action_type") or "[missing]",
        "history_bucket": _bucket(len(history), kind="history"),
        "candidate_position_bucket": _bucket(candidates.index(target) + 1, kind="position"),
        "candidate_position": candidates.index(target) + 1,
        "last_event_gap_bucket": _bucket(last_event_gap, kind="gap"),
        "target_last_seen_gap_bucket": _bucket(now - target_last, kind="gap") if target_last is not None else "not_seen",
        "last_hotel_gap_bucket": _bucket(now - last_hotel_time, kind="gap") if last_hotel_time is not None else "no_hotel",
        "distinct_history_hotels": len(set(hotels)),
        "retained_history_length": len(history),
        "candidate_count": len(candidates),
    }


def _aggregate() -> dict:
    return {"total": 0, "pattern": Counter(), "clickout_pattern": Counter(), "last_action": Counter(),
            "history_bucket": Counter(), "candidate_position_bucket": Counter(),
            "last_event_gap_bucket": Counter(), "target_last_seen_gap_bucket": Counter(),
            "last_hotel_gap_bucket": Counter(), "distinct_hotel_bucket": Counter(),
            "candidate_count": Counter(), "cross": defaultdict(Counter), "sessions": set()}


def _update(acc: dict, result: dict, session_key: tuple[str, str]) -> None:
    acc["total"] += 1
    acc["sessions"].add(session_key)
    for key in ("pattern", "clickout_pattern", "last_action", "history_bucket",
                "candidate_position_bucket", "last_event_gap_bucket",
                "target_last_seen_gap_bucket", "last_hotel_gap_bucket"):
        acc[key][result[key]] += 1
    n = result["distinct_history_hotels"]
    acc["distinct_hotel_bucket"]["0" if n == 0 else "1" if n == 1 else "2_5" if n <= 5 else "6_plus"] += 1
    acc["candidate_count"][str(result["candidate_count"])] += 1
    for key in ("clickout_pattern", "history_bucket", "last_action", "candidate_position_bucket",
                "last_event_gap_bucket"):
        acc["cross"][key][(result["pattern"], result[key])] += 1


def _render(acc: dict) -> dict:
    n = acc["total"]
    def rows(counts: Counter) -> dict:
        return {str(k): {"n": v, "pct": round(100 * v / n, 4) if n else 0.0}
                for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}
    return {"n_examples": n, "n_distinct_sessions": len(acc["sessions"]),
            **{key: rows(acc[key]) for key in (
                "pattern", "clickout_pattern", "last_action", "history_bucket",
                "candidate_position_bucket", "last_event_gap_bucket", "target_last_seen_gap_bucket",
                "last_hotel_gap_bucket", "distinct_hotel_bucket", "candidate_count")},
            "cross": {key: {p: {v: c for (pat, v), c in sorted(table.items()) if pat == p}
                            for p in PATTERNS} for key, table in acc["cross"].items()}}


def audit(path: str | Path, *, max_examples: int | None = None) -> dict:
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive")
    totals = {key: _aggregate() for key in ("all", "train", "validation", "test")}
    for i, example in enumerate(read_examples(path)):
        if max_examples is not None and i >= max_examples:
            break
        result = classify(example)
        split = split_of(int(example["target_timestamp"]))
        identity = (str(example["user_id"]), str(example["session_id"]))
        _update(totals["all"], result, identity)
        _update(totals[split], result, identity)
    return {
        "protocol": {"unit": "emitted next-clickout example", "history_window": "retained 1..50 events",
                     "ordering": "inherits upstream (timestamp,step,source-row) sort",
                     "hotel_actions_provisional": sorted(HOTEL_ACTIONS),
                     "splits_utc": {"train": "before 2018-11-05", "validation": "2018-11-05", "test": "2018-11-06 onward"},
                     "sample_limit": max_examples},
        "groups": {key: _render(value) for key, value in totals.items()},
        "caveats": ["Clickout-row candidate sets are not independently confirmed pre-click exposures.",
                    "Truncated history: unseen means unseen in retained history, not necessarily full session.",
                    "Search-for-item hotel-ID semantics remain provisional.",
                    "Examples in one session are correlated; counts are example weighted.",
                    "No raw zero-history clickouts are included; a separate raw audit is required."]}


def markdown(report: dict) -> str:
    lines = ["# Checkpoint A.2 — Behavioral pattern audit", "", "Generated from existing next-click-out JSONL; no baselines were trained.", ""]
    for name, group in report["groups"].items():
        lines.extend([f"## {name} ({group['n_examples']:,} examples; {group['n_distinct_sessions']:,} sessions)", "",
                      "| Hotel behavior | Count | Share of examples |", "|---|---:|---:|"])
        for category in PATTERNS:
            row = group["pattern"].get(category, {"n": 0, "pct": 0})
            lines.append(f"| {category} | {row['n']:,} | {row['pct']:.2f}% |")
        lines.append("")
    lines.extend(["## Caveats", ""] + [f"- {v}" for v in report["caveats"]])
    lines.extend(["", "See the JSON for action distributions, time gaps, candidate positions and pattern cross-tabs.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/next_clickout.jsonl")
    parser.add_argument("--output", default="data/processed/behavior_patterns_a2.json")
    parser.add_argument("--report", default="data/processed/behavior_patterns_a2.md")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    result = audit(args.input, max_examples=args.max_examples)
    for path, content in ((Path(args.output), json.dumps(result, indent=2)), (Path(args.report), markdown(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}")
    for split, group in result["groups"].items():
        print(f"[{split}] examples={group['n_examples']} sessions={group['n_distinct_sessions']}")
        for pattern in PATTERNS:
            item = group["pattern"].get(pattern, {"n": 0, "pct": 0})
            print(f"  {pattern:30s} n={item['n']:8d} share={item['pct']:.2f}%")


if __name__ == "__main__":
    main()
