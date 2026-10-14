"""Read-only audit of raw click-out histories and provisional item-reference semantics.

This is NOT a baseline model. Full raw CSV sorting may use substantial memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from travelerreclab.data.build_sequences import _items, _text

HOTEL_ACTIONS = {
    "clickout item", "interaction item deals", "interaction item image",
    "interaction item info", "interaction item rating", "search for item",
}
REQUIRED = ["user_id", "session_id", "timestamp", "step", "action_type", "reference", "impressions", "prices"]


def audit_frames(events: pd.DataFrame, metadata_ids: set[str], *, max_history: int = 50) -> dict:
    """Reconstruct full raw sessions, keeping the same ordering policy as build_sequences.

    Selection checks reproduce the *eligibility predicate* of build_sequences;
    output does not alter existing examples. Prefix-only inputs are not
    representative, and session counts may be partial in that mode.
    """
    if max_history < 1:
        raise ValueError("max_history must be positive")
    missing = set(REQUIRED) - set(events.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if events[["user_id", "session_id", "timestamp", "step"]].isna().any().any():
        raise ValueError("Identity, timestamp and step must be present")
    df = events.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise")
    df["step"] = pd.to_numeric(df["step"], errors="raise")
    df["_source_row"] = range(len(df))
    df = df.sort_values(
        ["user_id", "session_id", "timestamp", "step", "_source_row"], kind="stable"
    )
    out = Counter()
    search_ref = Counter()
    for _, session in df.groupby(["user_id", "session_id"], sort=False):
        out["sessions"] += 1
        history: list[tuple[str | None, str | None]] = []
        earlier_clickouts: list[str] = []
        for row in session.itertuples(index=False):
            action = _text(row.action_type)
            ref = _text(row.reference)
            if action == "search for item":
                search_ref["total"] += 1
                if ref is not None:
                    search_ref["nonempty"] += 1
                    if ref in metadata_ids:
                        search_ref["in_metadata"] += 1
                else:
                    search_ref["missing"] += 1
            if action == "clickout item":
                out["raw_clickouts"] += 1
                out["first_clickout" if not earlier_clickouts else
                    "repeat_clickout" if ref is not None and earlier_clickouts[-1] == ref
                    else "switch_clickout"] += 1
                if len(history) == 0:
                    out["zero_history"] += 1
                else:
                    out["has_history"] += 1
                    if len(history) == max_history:
                        out["exactly_max_history_before_target"] += 1
                    elif len(history) > max_history:
                        out["truncated_if_emitted"] += 1
                    else:
                        out["below_max_history_before_target"] += 1
                    if ref is not None:
                        candidates = _items(row.impressions)
                        prices = _items(row.prices)
                        if candidates and ref in candidates and len(candidates) == len(prices):
                            out["eligible_min_history_1"] += 1
                            if len(history) > max_history:
                                out["eligible_genuinely_truncated"] += 1
                            # How often does the target appear in the full history but
                            # not the retained 50 events? Only hotel-typed actions count.
                            hotel_refs = [r for a, r in history if a in HOTEL_ACTIONS and r]
                            retained_refs = [r for a, r in history[-max_history:]
                                             if a in HOTEL_ACTIONS and r]
                            if ref in hotel_refs and ref not in retained_refs:
                                out["eligible_target_lost_by_truncation"] += 1
                            if ref in hotel_refs:
                                out["eligible_target_seen_full_history"] += 1
                if ref is not None:
                    earlier_clickouts.append(ref)
            history.append((action, ref))
    return {
        "protocol": {
            "unit": "raw click-out events; search-for-item actions separately",
            "ordering": "(user_id,session_id,timestamp,step,source-row)",
            "max_history": max_history,
            "metadata_membership": "reference string equality with item_metadata.item_id",
            "limitations": [
                "Metadata membership supports but does not prove semantic intent of search-for-item.",
                "First/repeat/switch here use all preceding clickouts in the raw session.",
                "Eligibility matches the sequence builder's input predicate for min_history=1.",
                "Prefix-row samples may cut sessions and are not representative.",
                "Clickout-row candidates do not establish independently logged prior exposure.",
            ],
        },
        "raw": dict(sorted(out.items())),
        "search_for_item": dict(sorted(search_ref.items())),
        "checks": {
            "clickout_partition_matches": sum(out[k] for k in
                ("first_clickout", "repeat_clickout", "switch_clickout")) == out["raw_clickouts"],
            "history_partition_matches": out["zero_history"] + out["has_history"] == out["raw_clickouts"],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw/train.csv")
    p.add_argument("--metadata", default="data/raw/item_metadata.csv")
    p.add_argument("--output", default="data/processed/raw_semantics_a3.json")
    p.add_argument("--max-rows", type=int, help="Prefix-only smoke test (not representative)")
    p.add_argument("--max-history", type=int, default=50)
    a = p.parse_args()
    if a.max_rows is not None and a.max_rows < 1:
        p.error("--max-rows must be positive")
    metadata = pd.read_csv(a.metadata, usecols=["item_id"], dtype={"item_id": "string"})
    ids = set(metadata["item_id"].dropna().astype(str))
    df = pd.read_csv(a.input, usecols=REQUIRED, nrows=a.max_rows,
                     dtype={"reference": "string", "impressions": "string", "prices": "string"})
    result = audit_frames(df, ids, max_history=a.max_history)
    result["protocol"]["max_rows_prefix"] = a.max_rows
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")
    for key, count in result["raw"].items():
        print(f"  {key:40s} {count:>10}")
    print("Search-for-item:", result["search_for_item"])
    print("Checks:", result["checks"])


if __name__ == "__main__":
    main()
