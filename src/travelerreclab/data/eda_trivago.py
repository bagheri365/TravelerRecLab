"""Streaming, reproducible EDA for the Trivago RecSys 2019 training CSV.

No dataset is downloaded; no event ordering is assumed for streaming chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

import pandas as pd

COLUMNS = [
    "user_id", "session_id", "timestamp", "step", "action_type", "reference",
    "platform", "city", "device", "current_filters", "impressions", "prices",
]


def _nonnull(value: object) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def _pipe_tokens(value: object) -> list[str]:
    return str(value).split("|") if _nonnull(value) else []


def _quantiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {}
    series = pd.Series(values)
    return {"min": int(series.min()), "p50": float(series.quantile(.5)),
            "p90": float(series.quantile(.9)), "p99": float(series.quantile(.99)),
            "max": int(series.max())}


def audit(path: str | Path, chunksize: int = 250_000) -> dict:
    """Analyze the entire file in chunks; keeps user/session aggregates in memory."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    rows = 0
    action_counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    action_missing: dict[str, Counter[str]] = defaultdict(Counter)
    session_counts: Counter[tuple[str, str]] = Counter()
    user_sessions: dict[str, set[str]] = defaultdict(set)
    session_clicks: Counter[tuple[str, str]] = Counter()
    clicks = Counter()
    impression_sizes: Counter[int] = Counter()
    click_steps: list[int] = []
    min_ts: int | None = None
    max_ts: int | None = None
    actions_by_day: Counter[str] = Counter()
    for part in pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=True):
        missing_columns = set(COLUMNS) - set(part.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        rows += len(part)
        action_counts.update(part["action_type"].fillna("<missing>").value_counts().to_dict())
        for col in COLUMNS:
            missing[col] += int(part[col].isna().sum() + part[col].fillna("").eq("").sum() - part[col].isna().sum())
        for action, subset in part.groupby("action_type", dropna=False):
            key = str(action)
            for col in ("reference", "current_filters", "impressions", "prices"):
                action_missing[key][col] += int(subset[col].isna().sum() + subset[col].fillna("").eq("").sum() - subset[col].isna().sum())
        keys = part[["user_id", "session_id"]].fillna("<missing>")
        counts = keys.value_counts(sort=False)
        for (user, session), count in counts.items():
            session_counts[(str(user), str(session))] += int(count)
            user_sessions[str(user)].add(str(session))
        ts = pd.to_numeric(part["timestamp"], errors="coerce").dropna()
        if not ts.empty:
            lo, hi = int(ts.min()), int(ts.max())
            min_ts = lo if min_ts is None else min(min_ts, lo)
            max_ts = hi if max_ts is None else max(max_ts, hi)
            days = pd.to_datetime(ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")
            actions_by_day.update(days.value_counts().to_dict())
        click_rows = part.loc[part["action_type"] == "clickout item"]
        clicks["total"] += len(click_rows)
        for row in click_rows.itertuples(index=False):
            key = (str(row.user_id), str(row.session_id))
            session_clicks[key] += 1
            try:
                click_steps.append(int(row.step))
            except (TypeError, ValueError):
                clicks["invalid_step"] += 1
            candidates = _pipe_tokens(row.impressions)
            prices = _pipe_tokens(row.prices)
            if not candidates:
                clicks["missing_candidates"] += 1
                continue
            clicks["with_candidates"] += 1
            impression_sizes[len(candidates)] += 1
            if _nonnull(row.reference) and str(row.reference) in candidates:
                clicks["target_in_candidates"] += 1
            if len(prices) != len(candidates):
                clicks["price_length_mismatch"] += 1
            if len(set(candidates)) != len(candidates):
                clicks["duplicate_candidate_items"] += 1
    click_sessions = sum(v > 0 for v in session_clicks.values())
    return {
        "source": str(path), "rows": rows, "users": len(user_sessions),
        "sessions": len(session_counts), "clickout_sessions": click_sessions,
        "timestamp_min_utc": pd.to_datetime(min_ts, unit="s", utc=True).isoformat() if min_ts is not None else None,
        "timestamp_max_utc": pd.to_datetime(max_ts, unit="s", utc=True).isoformat() if max_ts is not None else None,
        "action_counts": dict(action_counts.most_common()),
        "missing_counts": dict(missing),
        "missing_by_action": {a: dict(c) for a, c in action_missing.items()},
        "events_per_session": _quantiles(list(session_counts.values())),
        "sessions_per_user": _quantiles([len(s) for s in user_sessions.values()]),
        "clickouts_per_clickout_session": _quantiles([x for x in session_clicks.values() if x]),
        "clickout_step": _quantiles(click_steps),
        "clickout_checks": dict(clicks),
        "candidate_count": _quantiles(list(impression_sizes.elements())),
        "daily_event_counts": dict(sorted(actions_by_day.items())),
        "methodology": [
            "Full-file descriptive scan; in-memory unique user/session keys scale with their cardinality.",
            "Clickout impression sets are recorded on clickout rows; availability before clickout is not established.",
            "Clickout step distribution is NOT a verified causal prefix-length distribution.",
            "Source event ordering and data leakage require separate targeted validation.",
        ],
    }


def render_markdown(stats: dict) -> str:
    n = stats["rows"]
    lines = ["# Trivago EDA — observed results", "",
             "> Auto-generated. Do not confuse data observations with model choices.", "",
             f"- Rows: **{n:,}**", f"- Users: **{stats['users']:,}**",
             f"- User/session pairs: **{stats['sessions']:,}**",
             f"- Sessions with click-outs: **{stats['clickout_sessions']:,}**",
             f"- Event range (UTC): {stats['timestamp_min_utc']} to {stats['timestamp_max_utc']}",
             "", "## Action distribution", "", "| Action | Count | Share |", "|---|---:|---:|"]
    for action, count in stats["action_counts"].items():
        lines.append(f"| {action} | {count:,} | {count/n:.2%} |" if n else f"| {action} | 0 | n/a |")
    lines += ["", "## Click-out data quality", ""]
    checks = stats["clickout_checks"]
    for key in ("total", "with_candidates", "target_in_candidates", "price_length_mismatch",
                "duplicate_candidate_items", "missing_candidates", "invalid_step"):
        lines.append(f"- {key}: {checks.get(key, 0):,}")
    lines += ["", "## Distributions", ""]
    for key in ("events_per_session", "sessions_per_user", "clickouts_per_clickout_session",
                "clickout_step", "candidate_count"):
        lines.append(f"- **{key}**: `{stats[key]}`")
    lines += ["", "## Interpretation guardrails", ""]
    lines += [f"- {line}" for line in stats["methodology"]]
    lines += ["", "## Decision questions before v0.2", "",
              "1. Is a click-out target included in the displayed candidate set? How often?",
              "2. Is `step` ordered and unique inside (user_id, session_id)? Validate separately.",
              "3. What fraction of click-outs have valid causal histories, after building prefixes?",
              "4. Do action mix and sequence length change for click-out sessions?",
              "5. Which time-based split avoids leakage across repeated user sessions?",
              "6. Should we predict next click-out globally or rerank the recorded candidate list?", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--output-dir", default="data/processed/eda")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    result = audit(args.input, chunksize=args.chunksize)
    (dest / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (dest / "report.md").write_text(render_markdown(result))
    print(f"Analyzed {result['rows']:,} events; wrote {dest / 'report.md'} and summary.json")


if __name__ == "__main__":
    main()
