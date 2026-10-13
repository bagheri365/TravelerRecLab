"""Inspect a Trivago RecSys 2019 event CSV without modifying it."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = frozenset(
    {"user_id", "session_id", "timestamp", "step", "action_type", "reference", "impressions", "prices"}
)


def summarize_events(path: str | Path) -> dict:
    """Read an event CSV and return schema/quality diagnostics.

    The full CSV is loaded for v0.0; chunked auditing can be added if memory becomes a problem.
    Session identity uses (user_id, session_id), not session_id alone.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}; download the Trivago data first.")
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required Trivago columns: {', '.join(sorted(missing))}")
    session_sizes = df.groupby(["user_id", "session_id"], dropna=False).size()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_fraction": df.isna().mean().round(4).to_dict(),
        "actions": df["action_type"].value_counts(dropna=False).to_dict(),
        "users": int(df["user_id"].nunique(dropna=False)),
        "sessions": int(len(session_sizes)),
        "events_per_session": session_sizes.describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/raw/train.csv", help="Path to Trivago train.csv")
    args = parser.parse_args()
    results = summarize_events(args.path)
    for name, value in results.items():
        print(f"\n{name}: {value}")


if __name__ == "__main__":
    main()
