"""Final pre-baseline audit: ordering reconciliation and metadata coverage.

This tool never claims click-out impression lists were observed before the click.
The ordering identity is exact for min_history=1 and no other filters besides
those in build_sequences.build_examples; the full CSV is streamed in chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from travelerreclab.data.validate_raw import split_list

COLS = ["user_id", "session_id", "timestamp", "step", "action_type", "reference", "impressions", "prices"]


def eligible_clickout(row: object) -> bool:
    """Check only the candidate predicates, not minimum history length."""
    if row.action_type != "clickout item" or pd.isna(row.reference):
        return False
    ref = str(row.reference).strip()
    candidates = split_list(row.impressions)
    prices = split_list(row.prices)
    return bool(ref and candidates and ref in candidates and len(candidates) == len(prices))


def audit_first_events(csv_path: str | Path, chunksize: int = 250_000) -> dict:
    """Explain eligible-count difference between source and timestamp/step sort.

    For min_history=1, only the first event of each session can fail the
    history rule. All other valid click-outs qualify. We can therefore compare
    the eligibility of the source-first and sort-first rows per session without
    storing complete event histories or doing an expensive global sort.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    first: dict[tuple[str, str], tuple[int, int, int, bool, str]] = {}
    sorted_first: dict[tuple[str, str], tuple[int, int, int, bool, str]] = {}
    event_count: Counter = Counter()
    malformed = 0
    position = 0
    for chunk in pd.read_csv(csv_path, usecols=COLS, chunksize=chunksize,
                             dtype={"user_id": "string", "session_id": "string", "reference": "string",
                                    "impressions": "string", "prices": "string"}):
        for r in chunk.itertuples(index=False):
            if pd.isna(r.user_id) or pd.isna(r.session_id) or pd.isna(r.timestamp) or pd.isna(r.step):
                malformed += 1
                position += 1
                continue
            key = (str(r.user_id), str(r.session_id))
            candidate_valid = eligible_clickout(r)
            item = (int(r.timestamp), int(r.step), position, candidate_valid, str(r.action_type))
            if key not in first:
                first[key] = item
            if key not in sorted_first or item[:3] < sorted_first[key][:3]:
                sorted_first[key] = item
            event_count[key] += 1
            position += 1
    changed = []
    difference = 0
    source_excluded = 0
    sorted_excluded = 0
    for key, source in first.items():
        earliest = sorted_first[key]
        source_excluded += int(source[3])
        sorted_excluded += int(earliest[3])
        delta = int(source[3]) - int(earliest[3])
        difference += delta
        if source[2] != earliest[2] and (delta or len(changed) < 30):
            changed.append({"user_id": key[0], "session_id": key[1], "events": event_count[key],
                            "source_first": {"timestamp": source[0], "step": source[1], "row_number_zero_based": source[2], "valid_clickout": source[3]},
                            "sort_first": {"timestamp": earliest[0], "step": earliest[1], "row_number_zero_based": earliest[2], "valid_clickout": earliest[3]},
                            "delta_sorted_minus_source": delta})
    return {"sessions": len(first), "invalid_identity_or_order_rows": malformed,
            "source_first_valid_clickouts": source_excluded,
            "sorted_first_valid_clickouts": sorted_excluded,
            "predicted_sorted_minus_source_eligible": difference,
            "sessions_with_different_first_event": sum(first[k][2] != sorted_first[k][2] for k in first),
            "sample_changed_sessions": changed,
            "interpretation": "Exact for min_history=1 when all otherwise-valid clickouts are retained; does not decide whether timestamp-first or step-first ordering is more truthful."}


def audit_item_coverage(events_path: str | Path, metadata_path: str | Path, chunksize: int = 250_000) -> dict:
    """Measure item metadata presence, action-level click popularity and tags."""
    clicked: Counter = Counter()
    for chunk in pd.read_csv(events_path, usecols=["action_type", "reference"], chunksize=chunksize,
                             dtype={"reference": "string"}):
        refs = chunk.loc[chunk["action_type"] == "clickout item", "reference"].dropna().str.strip()
        clicked.update(x for x in refs if x)
    ids: set[str] = set()
    props_seen = Counter()
    missing_properties = 0
    duplicates = 0
    rows = 0
    for chunk in pd.read_csv(metadata_path, chunksize=chunksize, dtype="string"):
        if not {"item_id", "properties"}.issubset(chunk.columns):
            raise ValueError("Metadata must have item_id and properties")
        for r in chunk.itertuples(index=False):
            rows += 1
            if pd.isna(r.item_id):
                continue
            hotel = str(r.item_id).strip()
            if hotel in ids:
                duplicates += 1
            ids.add(hotel)
            if pd.isna(r.properties) or not str(r.properties).strip():
                missing_properties += 1
            else:
                props_seen.update(p.strip() for p in str(r.properties).split("|") if p.strip())
    matched = clicked.keys() & ids
    total_clickouts = sum(clicked.values())
    covered_clickouts = sum(clicked[item] for item in matched)
    values = sorted(clicked.values(), reverse=True)
    k = max(1, (len(values) + 99) // 100) if values else 0
    return {"metadata_rows": rows, "metadata_unique_items": len(ids), "duplicate_metadata_ids": duplicates,
            "metadata_rows_missing_properties": missing_properties,
            "distinct_clicked_items": len(clicked), "clicked_items_with_metadata": len(matched),
            "distinct_clicked_item_coverage": len(matched) / len(clicked) if clicked else None,
            "total_clickouts": total_clickouts, "clickouts_on_items_with_metadata": covered_clickouts,
            "clickout_weighted_metadata_coverage": covered_clickouts / total_clickouts if total_clickouts else None,
            "top_1pct_clicked_items_clickout_share": sum(values[:k]) / total_clickouts if total_clickouts else None,
            "top_clicked_items": clicked.most_common(10), "top_metadata_properties": props_seen.most_common(20),
            "caution": "Property tags are metadata descriptions, not proof that features were available at every historical prediction timestamp."}


def run(events: str | Path, metadata: str | Path, output_dir: str | Path,
        chunksize: int = 250_000, expected_delta: int = 3) -> dict:
    ordered = audit_first_events(events, chunksize)
    coverage = audit_item_coverage(events, metadata, chunksize)
    result = {"ordering": ordered, "metadata_and_popularity": coverage,
              "expected_sorted_minus_source_eligible": expected_delta,
              "difference_matches_reported": ordered["predicted_sorted_minus_source_eligible"] == expected_delta}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# v0.1.8 — Ordering & metadata evidence", "",
             "This is an audit, not a model-performance result.", "",
             "## Eligibility discrepancy", "",
             f"- Sorted minus source-order predicted eligibility: **{ordered['predicted_sorted_minus_source_eligible']}**",
             f"- Matches earlier observed difference ({expected_delta}): **{result['difference_matches_reported']}**",
             f"- Sessions where first event differs: **{ordered['sessions_with_different_first_event']:,}**",
             f"- Invalid order/identity rows: **{ordered['invalid_identity_or_order_rows']:,}**", "",
             "Inspect `sample_changed_sessions` in summary.json before selecting ordering policy.", "",
             "## Item metadata and click concentration", "",
             f"- Distinct clicked IDs with metadata: **{coverage['clicked_items_with_metadata']:,} / {coverage['distinct_clicked_items']:,}**",
             f"- Distinct-clicked-item metadata coverage: **{coverage['distinct_clicked_item_coverage']}**",
             f"- Clickout-weighted metadata coverage: **{coverage['clickout_weighted_metadata_coverage']}**",
             f"- Top 1% clicked items account for share **{coverage['top_1pct_clicked_items_clickout_share']}** of clickouts.", "",
             "## Interpretation guardrails", "",
             "- The reconciliation applies to `min_history=1` and the exact existing generator predicates.",
             "- A mismatch means investigate event sorting and parsing; do not silently force counts to agree.",
             "- Metadata tags can help new-item generalization, but their historical availability is not verified.",
             "- Recorded clickout impressions are not independently timestamped pre-click exposure logs.",
             "- Do not commit raw event data or the generated summary, which can contain anonymized IDs."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--metadata", default="data/raw/item_metadata.csv")
    parser.add_argument("--output-dir", default="data/processed/final_eda")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--expected-delta", type=int, default=3)
    args = parser.parse_args()
    result = run(args.input, args.metadata, args.output_dir, args.chunksize, args.expected_delta)
    print(f"Report: {Path(args.output_dir) / 'report.md'}")
    print(f"Reconciled: {result['difference_matches_reported']}")


if __name__ == "__main__":
    main()
