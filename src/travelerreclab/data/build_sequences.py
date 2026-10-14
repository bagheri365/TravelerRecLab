"""Build causal, session-local next-click-out examples from Trivago events.

The target click-out itself is never part of the input history. Impression
lists are attached solely as *evaluation candidate sets*; because they are
logged on the click-out event, they must not be mistaken for independently
observed pre-decision impressions in an online ranking experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from travelerreclab.data.inspect_trivago import REQUIRED_COLUMNS

CLICKOUT = "clickout item"


def _text(value: object) -> str | None:
    if pd.isna(value):
        return None
    result = str(value).strip()
    return result if result else None


def _items(value: object) -> list[str]:
    raw = _text(value)
    return [item.strip() for item in raw.split("|") if item.strip()] if raw else []


def build_examples(
    events: pd.DataFrame,
    *,
    max_history: int = 50,
    min_history: int = 1,
) -> list[dict]:
    """Construct one example per eligible click-out using strictly earlier events.

    Group by (user_id, session_id) and sort by (timestamp, step, original row
    position). Events with a missing identity, timestamp, or step are rejected,
    rather than silently moved across time boundaries. Output is deterministic.
    """
    if max_history < 1 or min_history < 0 or min_history > max_history:
        raise ValueError("Require max_history >= 1 and 0 <= min_history <= max_history")
    missing = REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Missing required Trivago columns: {', '.join(sorted(missing))}")
    if events[list(["user_id", "session_id", "timestamp", "step"])].isna().any().any():
        raise ValueError("Identity, timestamp, and step must not be null")

    df = events.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise")
    df["step"] = pd.to_numeric(df["step"], errors="raise")
    if not df["timestamp"].notna().all() or not df["step"].notna().all():
        raise ValueError("Invalid timestamp or step")
    df["_input_order"] = range(len(df))
    df = df.sort_values(
        ["user_id", "session_id", "timestamp", "step", "_input_order"], kind="stable"
    )
    examples = []
    for (user_id, session_id), group in df.groupby(["user_id", "session_id"], sort=False):
        history: list[dict] = []
        for row in group.to_dict("records"):
            action = _text(row["action_type"])
            reference = _text(row["reference"])
            if action == CLICKOUT and reference is not None and len(history) >= min_history:
                candidates = _items(row["impressions"])
                prices = _items(row["prices"])
                # Click-out impression arrays are provided at target time, not
                # invented for history; malformed arrays are excluded.
                if candidates and reference in candidates and len(prices) == len(candidates):
                    examples.append(
                        {
                            "user_id": str(user_id),
                            "session_id": str(session_id),
                            "target_timestamp": int(row["timestamp"]),
                            "target_step": int(row["step"]),
                            "target_item": reference,
                            "candidate_items": candidates,
                            "candidate_prices": prices,
                            "history": history[-max_history:].copy(),
                        }
                    )
            history.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "step": int(row["step"]),
                    "action_type": action,
                    "reference": reference,
                }
            )
    return examples


def build_file(input_path: str | Path, output_path: str | Path, **kwargs: int) -> int:
    """Read Trivago CSV and write JSONL examples; return number of examples."""
    df = pd.read_csv(input_path, dtype={"reference": "string", "impressions": "string", "prices": "string"})
    examples = build_examples(df, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--output", default="data/processed/next_clickout.jsonl")
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=1)
    args = parser.parse_args()
    count = build_file(args.input, args.output, max_history=args.max_history, min_history=args.min_history)
    print(f"Wrote {count} next-click-out examples to {args.output}")


if __name__ == "__main__":
    main()
