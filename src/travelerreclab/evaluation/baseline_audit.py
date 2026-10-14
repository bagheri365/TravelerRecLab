"""Position controls, tie-neutral metrics, slices and validation-only decay selection.

Runs on v0.1 causal JSONL. Candidate sets are clickout-row observations,
not verified independently timestamped pre-click impressions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from .baselines import (HOTEL_ACTIONS, fit_popularity, read_examples, score_candidates,
                        split_of, valid_example)

MODELS = ("position", "popularity", "last_item", "action_weighted", "recency_weighted")
SEGMENTS = ("all", "target_seen", "target_unseen", "history_1_5", "history_6_20", "history_21_plus")


class RankingMetrics:
    """Both logged-order ranking and expected metrics under random tie permutations."""
    def __init__(self):
        self.n = 0
        self.logged_mrr = 0.0
        self.neutral_mrr = 0.0
        self.logged_r5 = 0.0
        self.neutral_r5 = 0.0
        self.logged_ndcg5 = 0.0
        self.neutral_ndcg5 = 0.0

    def add(self, scores: list[float], target_index: int):
        score = scores[target_index]
        ahead = sum(s > score for s in scores)
        tied_before = sum(s == score for s in scores[:target_index])
        ties = sum(s == score for s in scores)
        rank = ahead + tied_before + 1
        possible = range(ahead + 1, ahead + ties + 1)
        self.n += 1
        self.logged_mrr += 1 / rank
        self.neutral_mrr += sum(1 / r for r in possible) / ties
        self.logged_r5 += rank <= 5
        self.neutral_r5 += sum(r <= 5 for r in possible) / ties
        self.logged_ndcg5 += 1 / math.log2(rank + 1) if rank <= 5 else 0
        self.neutral_ndcg5 += sum(1 / math.log2(r + 1) if r <= 5 else 0 for r in possible) / ties

    def result(self):
        if not self.n:
            return {"n": 0, "mrr_logged_ties": None, "mrr_neutral_ties": None,
                    "recall@5_logged_ties": None, "recall@5_neutral_ties": None,
                    "ndcg@5_logged_ties": None, "ndcg@5_neutral_ties": None}
        return {"n": self.n, "mrr_logged_ties": self.logged_mrr / self.n,
                "mrr_neutral_ties": self.neutral_mrr / self.n,
                "recall@5_logged_ties": self.logged_r5 / self.n,
                "recall@5_neutral_ties": self.neutral_r5 / self.n,
                "ndcg@5_logged_ties": self.logged_ndcg5 / self.n,
                "ndcg@5_neutral_ties": self.neutral_ndcg5 / self.n}


def segments(e: dict) -> tuple[str, ...]:
    history = e.get("history", [])
    seen = {str(h["reference"]) for h in history
            if h.get("action_type") in HOTEL_ACTIONS and h.get("reference")}
    length = len(history)
    bucket = "history_1_5" if length <= 5 else "history_6_20" if length <= 20 else "history_21_plus"
    return ("all", "target_seen" if str(e["target_item"]) in seen else "target_unseen", bucket)


def compute(path: str | Path, *, val_day="2018-11-05", test_day="2018-11-06",
            decay_candidates=(300.0, 3600.0, 86400.0), max_examples=None):
    if not decay_candidates or any(x <= 0 for x in decay_candidates):
        raise ValueError("Positive decay candidates required")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    pop = fit_popularity(path, val_day=val_day, test_day=test_day)
    configs = [f"recency_{int(x)}s" for x in decay_candidates]
    names = (*MODELS[:-1], *configs)
    results = {split: {name: {s: RankingMetrics() for s in SEGMENTS}
                       for name in names} for split in ("validation", "test")}
    n = Counter()
    for e in read_examples(path):
        split = split_of(e["target_timestamp"], val_day=val_day, test_day=test_day)
        if split == "train" or not valid_example(e):
            continue
        if max_examples is not None and n[split] >= max_examples:
            continue
        n[split] += 1
        candidates = e["candidate_items"]
        idx = candidates.index(e["target_item"])
        scores = {
            "position": [-float(i) for i in range(len(candidates))],
            "popularity": score_candidates(e, "popularity", pop),
            "last_item": score_candidates(e, "last_item", pop),
            "action_weighted": score_candidates(e, "action_weighted", pop),
        }
        for decay, label in zip(decay_candidates, configs):
            scores[label] = score_candidates(e, "recency_weighted", pop, decay_seconds=decay)
        slices = segments(e)
        for name, values in scores.items():
            for seg in slices:
                results[split][name][seg].add(values, idx)

    # Choose decay on validation only. Reporting all test sweeps is explicitly
    # diagnostic; do not select using test results.
    winner = max(configs, key=lambda c: results["validation"][c]["all"].result()["mrr_neutral_ties"] or -1)
    return {
        "configuration": {"validation_start_utc": val_day, "test_start_utc": test_day,
                          "decay_candidates_seconds": list(decay_candidates),
                          "max_examples_per_split": max_examples},
        "selection": {"rule": "maximum validation all-segment tie-neutral MRR",
                      "selected_recency": winner},
        "sample_counts": dict(n),
        "metrics": {split: {name: {seg: meter.result() for seg, meter in sliced.items()}
                            for name, sliced in models.items()}
                    for split, models in results.items()},
        "caveats": [
            "The position baseline is the recorded click-out-row candidate order.",
            "Neutral tie handling averages metrics over all placements inside the tied score block.",
            "Validation chooses recency decay; test must not influence selection.",
            "This is click-out selection, not booking propensity or verified pre-click exposure.",
            "History-eligible examples exclude zero-history traffic.",
            "Action-weight tuning and confidence intervals are not included in this audit.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/next_clickout.jsonl")
    p.add_argument("--output", default="data/processed/baselines_v021_audit.json")
    p.add_argument("--max-examples", type=int)
    p.add_argument("--decays", type=float, nargs="+", default=[300, 3600, 86400])
    args = p.parse_args()
    result = compute(args.input, decay_candidates=tuple(args.decays), max_examples=args.max_examples)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")
    print(f"Selected by validation: {result['selection']['selected_recency']}")
    for split, models in result["metrics"].items():
        print(f"[{split}]")
        for name, slices in models.items():
            m = slices["all"]
            print(f"  {name:20s} n={m['n']:>7} logged={m['mrr_logged_ties']:.4f} neutral={m['mrr_neutral_ties']:.4f}")


if __name__ == "__main__":
    main()
