"""Validation-selected position + last-hotel control for next-click-out ranking.

Position is taken from the candidate array logged on the *target click-out row*;
this is an offline diagnostic, not a confirmed pre-click serving feature.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .baseline_audit import RankingMetrics
from .baselines import HOTEL_ACTIONS, read_examples, score_candidates, split_of, valid_example

DEFAULT_BOOSTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
PATTERNS = ("immediate_continuation", "return_to_earlier_hotel",
            "new_hotel_after_exploration", "no_prior_hotel_reference")
SEGMENTS = ("all", *PATTERNS)


def behavioral_pattern(example: dict) -> str:
    """Mutually exclusive categories based strictly on retained hotel-typed history."""
    refs = [str(h["reference"]) for h in example.get("history", [])
            if h.get("action_type") in HOTEL_ACTIONS and h.get("reference")]
    if not refs:
        return "no_prior_hotel_reference"
    target = str(example["target_item"])
    if refs[-1] == target:
        return "immediate_continuation"
    if target in refs:
        return "return_to_earlier_hotel"
    return "new_hotel_after_exploration"


def hybrid_scores(example: dict, boost: float) -> list[float]:
    """Normalize upstream position to [0,1], add boost to matching last hotel.

    Position 1 receives 1 and the final candidate receives 0. Singleton
    candidate sets receive 1. The boost is non-negative and scale-invariant
    across different candidate list sizes.
    """
    if boost < 0:
        raise ValueError("boost must be nonnegative")
    n = len(example["candidate_items"])
    if n == 0:
        raise ValueError("empty candidate list")
    base = [1 - i / (n - 1) if n > 1 else 1.0 for i in range(n)]
    last = score_candidates(example, "last_item", Counter())
    return [p + boost * x for p, x in zip(base, last)]


def compute(path: str | Path, *, boosts=DEFAULT_BOOSTS,
            val_day="2018-11-05", test_day="2018-11-06", max_examples=None) -> dict:
    boosts = tuple(float(b) for b in boosts)
    if not boosts or len(set(boosts)) != len(boosts) or any(b < 0 for b in boosts):
        raise ValueError("boosts must be unique nonnegative values")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    if val_day >= test_day:
        raise ValueError("validation date must precede test date")
    labels = [f"boost_{b:g}" for b in boosts]
    meters = {s: {name: {seg: RankingMetrics() for seg in SEGMENTS} for name in labels}
              for s in ("validation", "test")}
    counts = Counter()
    for e in read_examples(path):
        split = split_of(e["target_timestamp"], val_day=val_day, test_day=test_day)
        if split == "train" or not valid_example(e):
            continue
        if max_examples is not None and counts[split] >= max_examples:
            continue
        counts[split] += 1
        seg = behavioral_pattern(e)
        target_index = e["candidate_items"].index(e["target_item"])
        for boost, label in zip(boosts, labels):
            scores = hybrid_scores(e, boost)
            meters[split][label]["all"].add(scores, target_index)
            meters[split][label][seg].add(scores, target_index)
    validation = meters["validation"]
    if not counts["validation"]:
        raise ValueError("No validation examples: cannot select boost")
    # Tie rule is explicit: choose the smallest boost among equal validation MRR.
    best = max(range(len(boosts)), key=lambda i: (
        validation[labels[i]]["all"].result()["mrr_neutral_ties"], -boosts[i]))
    return {
        "configuration": {"boost_candidates": list(boosts), "validation_start_utc": val_day,
                          "test_start_utc": test_day, "max_examples_per_split": max_examples,
                          "position_scaling": "1 - candidate_index / (candidate_count - 1)"},
        "selection": {"rule": "maximize validation all-example tie-neutral MRR; smallest boost on ties",
                      "selected_boost": boosts[best], "selected_model": labels[best]},
        "sample_counts": dict(counts),
        "metrics": {split: {name: {seg: meter.result() for seg, meter in segs.items()}
                            for name, segs in methods.items()}
                    for split, methods in meters.items()},
        "limitations": [
            "Position comes from impressions on target click-out rows; pre-click availability is unverified.",
            "This is a candidate-reranking diagnostic, not a pure traveler representation baseline.",
            "History means the retained 1..50 earlier events, not the full session.",
            "The action list for hotel references includes provisionally typed 'search for item'.",
            "Boost is selected on validation only; inspect test once for the selected setting.",
            "Multiple examples per session are correlated; no clustered confidence intervals here.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/next_clickout.jsonl")
    parser.add_argument("--output", default="data/processed/hybrid_position_v023.json")
    parser.add_argument("--boosts", type=float, nargs="+", default=list(DEFAULT_BOOSTS))
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    result = compute(args.input, boosts=args.boosts, max_examples=args.max_examples)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    chosen = result["selection"]["selected_model"]
    print(f"Report: {out}")
    print(f"Selected on validation: {chosen}")
    for split in ("validation", "test"):
        print(f"[{split}]")
        for seg in SEGMENTS:
            m = result["metrics"][split][chosen][seg]
            print(f"  {seg:28s} n={m['n']:>7} tie-neutral MRR={m['mrr_neutral_ties']:.4f}")


if __name__ == "__main__":
    main()
