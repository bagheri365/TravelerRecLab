"""Auditable source-order, click-out exclusion, and item-metadata diagnostics.

Counts exclusions under *source file order*. Compare to build_sequences output,
which sorts (timestamp, step, original row), before treating counts as exact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

import pandas as pd

REQUIRED = {"user_id", "session_id", "timestamp", "step", "action_type", "reference", "impressions", "prices"}


def split_list(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [v.strip() for v in str(value).split("|") if v.strip()]


def audit_events(path: str | Path, chunksize: int = 250_000, min_history: int = 1) -> dict:
    if chunksize < 1 or min_history < 0:
        raise ValueError("chunksize must be positive and min_history nonnegative")
    state: dict[tuple[str, str], tuple[int, int, int]] = {}
    seen_steps: dict[tuple[str, str], set[int]] = defaultdict(set)
    previous_key = None
    closed_keys: set[tuple[str, str]] = set()
    counts: Counter = Counter()
    top_clicked: Counter = Counter()
    distinct_clicked: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype={"user_id": "string", "session_id": "string", "reference": "string", "impressions": "string", "prices": "string"}):
        missing = REQUIRED.difference(chunk.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for row in chunk.itertuples(index=False):
            key = (str(row.user_id), str(row.session_id))
            ts, step = int(row.timestamp), int(row.step)
            counts["rows"] += 1
            if key != previous_key:
                if previous_key is not None:
                    closed_keys.add(previous_key)
                if key in closed_keys:
                    counts["reentered_session_rows"] += 1
                previous_key = key
            n, last_ts, last_step = state.get(key, (0, -1, -1))
            if n:
                if ts < last_ts:
                    counts["timestamp_reversals"] += 1
                if ts == last_ts:
                    counts["timestamp_ties"] += 1
                if step <= last_step:
                    counts["nonincreasing_steps"] += 1
            if step in seen_steps[key]:
                counts["duplicate_step_rows"] += 1
            seen_steps[key].add(step)
            if row.action_type == "clickout item":
                counts["clickouts"] += 1
                ref = None if pd.isna(row.reference) else str(row.reference).strip()
                cands = split_list(row.impressions)
                prices = split_list(row.prices)
                if ref:
                    top_clicked[ref] += 1
                    distinct_clicked.add(ref)
                # Mirror build_sequences eligibility predicates in documented priority order.
                if not ref:
                    counts["excluded_missing_target"] += 1
                elif n < min_history:
                    counts["excluded_short_history"] += 1
                elif not cands:
                    counts["excluded_no_candidates"] += 1
                elif ref not in cands:
                    counts["excluded_target_not_in_candidates"] += 1
                elif len(cands) != len(prices):
                    counts["excluded_price_mismatch"] += 1
                else:
                    counts["eligible_in_source_order"] += 1
            state[key] = n + 1, ts, step
    counts["user_session_pairs"] = len(state)
    counts["distinct_clicked_items"] = len(distinct_clicked)
    result = {"counts": dict(counts), "top_clicked_items": top_clicked.most_common(20),
              "methodology": [
                  "Streaming audit uses row order from raw CSV, not builder's timestamp/step sort.",
                  "Exclusion categories are exclusive and follow the build_sequences predicate order.",
                  "Compare eligible_in_source_order to emitted examples; if unequal, investigate sorting or parsing.",
                  "reentered_session_rows counts rows whose session key was observed after a different key, not distinct sessions.",
                  "Session step uniqueness is checked over (user_id, session_id).",
                  "No proof of candidate impression availability before click-out.",
              ]}
    return result


def audit_metadata(path: str | Path, clicked_items: list[str] | None = None) -> dict:
    # Metadata is much smaller than event logs in typical challenge releases.
    df = pd.read_csv(path, dtype="string")
    if "item_id" not in df.columns:
        raise ValueError("item_metadata.csv requires item_id")
    items = set(df["item_id"].dropna().astype(str))
    result = {"rows": len(df), "columns": list(df.columns),
              "unique_items": len(items), "duplicate_item_rows": int(df["item_id"].duplicated().sum()),
              "missing_fraction": {col: round(float(df[col].isna().mean()), 6) for col in df.columns}}
    if clicked_items is not None:
        clicked = set(map(str, clicked_items))
        result["clicked_item_metadata_coverage"] = round(len(clicked & items) / len(clicked), 6) if clicked else None
        result["clicked_items_without_metadata"] = len(clicked - items)
    return result


def run(input_path: str | Path, metadata_path: str | Path, output_dir: str | Path,
        chunksize: int = 250_000, emitted_examples: int | None = None) -> dict:
    raw = audit_events(input_path, chunksize=chunksize)
    # The top-20 IDs are insufficient for a full metadata coverage estimate.
    meta = audit_metadata(metadata_path)
    result = {"raw": raw, "metadata": meta, "emitted_examples_reference": emitted_examples}
    if emitted_examples is not None:
        result["emitted_minus_source_order_eligible"] = emitted_examples - raw["counts"].get("eligible_in_source_order", 0)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    c = raw["counts"]
    lines = ["# Trivago raw data validation — v0.1.7", "", "## Source-order and exclusion audit", "",
             "| Measure | Count |", "|---|---:|"]
    lines += [f"| {name} | {value:,} |" for name, value in sorted(c.items())]
    lines += ["", "## Metadata", "", f"Columns: `{meta['columns']}`", "",
              f"Rows: {meta['rows']:,}; distinct items: {meta['unique_items']:,}.", "",
              "## Interpretation and limits", ""]
    lines += [f"- {note}" for note in raw["methodology"]]
    lines += ["- Metadata presence is NOT item coverage until full clicked-ID membership is computed.",
              "- Raw-order eligibility is not guaranteed to equal builder output if order anomalies exist."]
    if emitted_examples is not None:
        lines += ["", f"Emitted-example reference: **{emitted_examples:,}**; difference vs source-order eligibility: **{result['emitted_minus_source_order_eligible']:,}**."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--metadata", default="data/raw/item_metadata.csv")
    parser.add_argument("--output-dir", default="data/processed/raw_validation")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--emitted-examples", type=int, default=1_276_216)
    args = parser.parse_args()
    result = run(args.input, args.metadata, args.output_dir, args.chunksize, args.emitted_examples)
    print(f"Report: {Path(args.output_dir) / 'report.md'}")
    print(json.dumps(result['raw']['counts'], indent=2))


if __name__ == "__main__":
    main()
