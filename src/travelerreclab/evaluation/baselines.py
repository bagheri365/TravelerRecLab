"""Chronological, candidate-set baselines for Trivago next-click-out examples.

The impression set comes from the *target click-out row*: an offline
reconstruction, not proof that this set was available before the decision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HOTEL_ACTIONS = {
    "interaction item image", "interaction item info", "interaction item rating",
    "interaction item deals", "clickout item", "search for item",
}
DEFAULT_WEIGHTS = {
    "interaction item image": 1.0,
    "interaction item info": 1.5,
    "interaction item rating": 1.5,
    "interaction item deals": 1.5,
    "clickout item": 2.0,
    "search for item": 1.0,
}
MODELS = ("random", "popularity", "last_item", "action_weighted", "recency_weighted")


def split_of(timestamp: int, *, val_day: str = "2018-11-05", test_day: str = "2018-11-06") -> str:
    """UTC target-time split; test starts at test_day, validation at val_day."""
    day = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).date().isoformat()
    if val_day >= test_day:
        raise ValueError("val_day must be earlier than test_day")
    return "train" if day < val_day else "validation" if day < test_day else "test"


def read_examples(path: str | Path):
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {number}") from e


def valid_example(e: dict) -> bool:
    candidates = e.get("candidate_items") or []
    return bool(candidates and len(set(candidates)) == len(candidates)
                and e.get("target_item") in candidates)


def fit_popularity(path: str | Path, *, val_day: str, test_day: str) -> Counter:
    """Count only training-period targets, never validation/test clicks."""
    popularity: Counter = Counter()
    for e in read_examples(path):
        if valid_example(e) and split_of(e["target_timestamp"], val_day=val_day, test_day=test_day) == "train":
            popularity[str(e["target_item"])] += 1
    return popularity


def score_candidates(e: dict, model: str, popularity: Counter, *, decay_seconds: float = 3600.0, seed: int = 42) -> list[float]:
    """Score all candidates. Stable input order is the tie breaker.

    Last item is the last *hotel-typed* reference, not a destination/POI/filter.
    Recency uses event timestamps and clamps negative gaps to zero.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model}")
    if decay_seconds <= 0:
        raise ValueError("decay_seconds must be positive")
    candidates = [str(x) for x in e["candidate_items"]]
    if model == "popularity":
        return [float(popularity.get(h, 0)) for h in candidates]
    if model == "random":
        # Independent of Python randomized hash seed; stable across runs.
        key = f'{seed}|{e.get("user_id")}|{e.get("session_id")}|{e["target_timestamp"]}|{e.get("target_step")}'
        return [int.from_bytes(hashlib.sha256(f"{key}|{i}".encode()).digest()[:8], "big") / (2**64)
                for i in range(len(candidates))]
    history = [h for h in e.get("history", []) if h.get("action_type") in HOTEL_ACTIONS and h.get("reference")]
    if model == "last_item":
        last = str(history[-1]["reference"]) if history else None
        return [1.0 if h == last else 0.0 for h in candidates]
    totals: dict[str, float] = defaultdict(float)
    now = int(e["target_timestamp"])
    for h in history:
        hotel = str(h["reference"])
        weight = DEFAULT_WEIGHTS.get(h["action_type"], 1.0)
        if model == "recency_weighted":
            weight *= math.exp(-max(0, now - int(h["timestamp"])) / decay_seconds)
        totals[hotel] += weight
    return [totals[h] for h in candidates]


def rank_of_target(e: dict, scores: list[float]) -> int:
    """1-indexed rank; scores descending, original candidate index on ties."""
    candidates = e["candidate_items"]
    if len(scores) != len(candidates):
        raise ValueError("Score/candidate lengths differ")
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i)).index(candidates.index(e["target_item"])) + 1


class Metrics:
    def __init__(self):
        self.n = 0
        self.mrr = 0.0
        self.recall5 = 0
        self.recall10 = 0
        self.ndcg5 = 0.0
        self.ndcg10 = 0.0

    def add(self, rank: int):
        self.n += 1
        self.mrr += 1.0 / rank
        self.recall5 += rank <= 5
        self.recall10 += rank <= 10
        self.ndcg5 += 1.0 / math.log2(rank + 1) if rank <= 5 else 0
        self.ndcg10 += 1.0 / math.log2(rank + 1) if rank <= 10 else 0

    def result(self) -> dict:
        if not self.n:
            return {"n": 0, "mrr": None, "recall@5": None, "recall@10": None,
                    "ndcg@5": None, "ndcg@10": None}
        return {"n": self.n, "mrr": self.mrr / self.n, "recall@5": self.recall5 / self.n,
                "recall@10": self.recall10 / self.n, "ndcg@5": self.ndcg5 / self.n,
                "ndcg@10": self.ndcg10 / self.n}


def evaluate(path: str | Path, *, val_day: str = "2018-11-05", test_day: str = "2018-11-06",
             decay_seconds: float = 3600.0, max_examples: int | None = None) -> dict:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    popularity = fit_popularity(path, val_day=val_day, test_day=test_day)
    reports: dict[str, dict[str, Metrics]] = {
        split: {name: Metrics() for name in MODELS} for split in ("validation", "test")
    }
    counts: Counter = Counter()
    for e in read_examples(path):
        split = split_of(e["target_timestamp"], val_day=val_day, test_day=test_day)
        if split == "train":
            continue
        if not valid_example(e):
            counts["invalid_candidates"] += 1
            continue
        if max_examples is not None and counts[split] >= max_examples:
            continue
        counts[split] += 1
        for model in MODELS:
            scores = score_candidates(e, model, popularity, decay_seconds=decay_seconds)
            reports[split][model].add(rank_of_target(e, scores))
    return {
        "configuration": {"val_day": val_day, "test_day": test_day,
                          "decay_seconds": decay_seconds, "max_examples_per_split": max_examples},
        "train_popularity_items": len(popularity),
        "counts": dict(counts),
        "metrics": {split: {model: meter.result() for model, meter in methods.items()}
                    for split, methods in reports.items()},
        "limitations": [
            "Candidate lists are logged at target click-out, not separately proven pre-click exposures.",
            "The history-eligible JSONL excludes zero-history click-outs.",
            "Historical prefixes can contain earlier events that cross split boundaries, but targets fit popularity only in train.",
            "Action weights and recency decay are initial heuristics; tune only on validation.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/next_clickout.jsonl")
    p.add_argument("--output", default="data/processed/baselines_v02.json")
    p.add_argument("--val-day", default="2018-11-05")
    p.add_argument("--test-day", default="2018-11-06")
    p.add_argument("--decay-seconds", type=float, default=3600.0)
    p.add_argument("--max-examples", type=int, default=None, help="Per split; for fast smoke runs")
    args = p.parse_args()
    result = evaluate(args.input, val_day=args.val_day, test_day=args.test_day,
                      decay_seconds=args.decay_seconds, max_examples=args.max_examples)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {out}")
    for split, methods in result["metrics"].items():
        print(f"[{split}]")
        for name, metrics in methods.items():
            print(f"  {name:18} n={metrics['n']:>7} MRR={metrics['mrr']}")


if __name__ == "__main__":
    main()
