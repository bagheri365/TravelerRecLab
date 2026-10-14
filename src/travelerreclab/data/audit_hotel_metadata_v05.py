"""v0.5 pre-model audit: hotel metadata and candidate coverage, no training.

All split membership uses the target click-out timestamp. Candidate lists are
attached to target rows and not verified pre-click exposure logs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from travelerreclab.evaluation.baselines import HOTEL_ACTIONS, read_examples, split_of, valid_example

SLICES = ("immediate_continuation", "return_to_earlier_hotel",
          "new_hotel_after_exploration", "no_prior_hotel_reference")


def history_hotels(example: dict) -> list[str]:
    return [str(event["reference"]) for event in example.get("history", [])
            if event.get("action_type") in HOTEL_ACTIONS
            and event.get("reference") not in (None, "")]


def behavior(example: dict, history: list[str]) -> str:
    target = str(example["target_item"])
    if not history:
        return "no_prior_hotel_reference"
    if history[-1] == target:
        return "immediate_continuation"
    if target in history:
        return "return_to_earlier_hotel"
    return "new_hotel_after_exploration"


def inspect_examples(path: str | Path):
    """Create train-only membership and observed evaluation coverage sets.

    Train candidate membership reproduces the *set* used by the min_count=1
    neural vocab (train history + candidates). Train target membership is a
    stricter signal; candidate occurrence is not a positive interaction.
    """
    vocab_items: set[str] = set()
    train_targets: set[str] = set()
    eval_items: set[str] = set()
    train_n = 0
    for e in read_examples(path):
        if not valid_example(e):
            continue
        split = split_of(e["target_timestamp"])
        if split == "train":
            train_n += 1
            vocab_items.update(map(str, e["candidate_items"]))
            vocab_items.update(history_hotels(e))
            train_targets.add(str(e["target_item"]))
        else:
            eval_items.update(map(str, e["candidate_items"]))
            eval_items.update(history_hotels(e))
    return vocab_items, train_targets, eval_items, train_n


def inspect_metadata(path: str | Path, relevant: set[str]):
    present: set[str] = set()
    nonempty_properties: set[str] = set()
    tags: Counter[str] = Counter()
    property_count_hist: Counter[int] = Counter()
    duplicates = 0
    total = 0
    blank_ids = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"item_id", "properties"}.issubset(reader.fieldnames or []):
            raise ValueError("item_metadata.csv must contain item_id and properties")
        for row in reader:
            total += 1
            hotel = (row.get("item_id") or "").strip()
            if not hotel:
                blank_ids += 1
                continue
            if hotel in present:
                duplicates += 1
                continue
            # Store only relevant IDs, rather than materializing all tag lists.
            if hotel in relevant:
                present.add(hotel)
                props = set(x.strip() for x in (row.get("properties") or "").split("|") if x.strip())
                if props:
                    nonempty_properties.add(hotel)
                tags.update(props)
                property_count_hist[len(props)] += 1
    return present, nonempty_properties, {
        "metadata_rows": total,
        "blank_item_id_rows": blank_ids,
        "relevant_ids_with_duplicate_rows": duplicates,
        "relevant_ids_with_metadata": len(present),
        "relevant_ids_with_nonempty_properties": len(nonempty_properties),
        "property_tag_vocabulary_size_relevant": len(tags),
        "top_property_tags_relevant": tags.most_common(30),
        "property_count_histogram_relevant": sorted(property_count_hist.items()),
    }


def summarize(path: str | Path, metadata: str | Path) -> dict:
    vocab, targets, eval_items, train_n = inspect_examples(path)
    present, nonempty, metadata_summary = inspect_metadata(metadata, vocab | eval_items)
    metrics = defaultdict(lambda: defaultdict(Counter))
    examples_by_split = Counter()
    for e in read_examples(path):
        if not valid_example(e):
            continue
        split = split_of(e["target_timestamp"])
        if split == "train":
            continue
        history = history_hotels(e)
        segment = behavior(e, history)
        target = str(e["target_item"])
        candidates = [str(c) for c in e["candidate_items"]]
        for label in ("all", segment):
            c = metrics[split][label]
            c["examples"] += 1
            c["candidate_slots"] += len(candidates)
            c["target_in_metadata"] += target in present
            c["target_with_properties"] += target in nonempty
            c["target_in_train_vocab"] += target in vocab
            c["target_in_train_targets"] += target in targets
            c["target_in_retained_history"] += target in history
            c["candidate_slots_in_metadata"] += sum(x in present for x in candidates)
            c["candidate_slots_with_properties"] += sum(x in nonempty for x in candidates)
            c["candidate_slots_in_train_vocab"] += sum(x in vocab for x in candidates)
            c["candidate_slots_in_train_targets"] += sum(x in targets for x in candidates)
        examples_by_split[split] += 1
    result = {}
    for split in ("validation", "test"):
        result[split] = {}
        for label in ("all", *SLICES):
            count = dict(metrics[split][label]); n = count.get("examples", 0)
            slots = count.get("candidate_slots", 0)
            count["rates"] = {
                key + "_rate": count.get(key, 0) / (slots if key.startswith("candidate_slots_") else n)
                if (slots if key.startswith("candidate_slots_") else n) else None
                for key in ("target_in_metadata", "target_with_properties", "target_in_train_vocab",
                            "target_in_train_targets", "target_in_retained_history",
                            "candidate_slots_in_metadata", "candidate_slots_with_properties",
                            "candidate_slots_in_train_vocab", "candidate_slots_in_train_targets")
            }
            result[split][label] = count
    return {
        "configuration": {"split_policy": "UTC target timestamp; train < Nov 5, validation Nov 5, test Nov 6 2018",
                          "train_vocab_membership": "IDs in train candidates or hotel-bearing train history; min_count=1",
                          "train_target_membership": "IDs clicked as target in train",
                          "metadata_join": "exact string item_id match; first matching row",
                          "history": "retained hotel-bearing events only"},
        "train_examples": train_n,
        "train_vocabulary_unique_ids": len(vocab),
        "train_target_unique_ids": len(targets),
        "metadata": metadata_summary,
        "by_split_and_behavior": result,
        "limitations": [
            "Candidate lists are taken from target click-out rows, not verified pre-click exposures.",
            "Metadata coverage is not evidence attributes existed at historical serving time.",
            "Metadata tag frequencies are for relevant unique IDs, not click-weighted counts.",
            "Train vocabulary includes candidate exposure, not just positive labels; min_count=1 only.",
            "No performance gains or cold-start improvements are claimed by this descriptive audit.",
            "Behavior classes use retained history and provisional hotel action typing.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/next_clickout.jsonl")
    p.add_argument("--metadata", default="data/raw/item_metadata.csv")
    p.add_argument("--output", default="data/processed/hotel_metadata_audit_v05.json")
    a = p.parse_args()
    report = summarize(a.input, a.metadata)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {out}")
    for split in ("validation", "test"):
        row = report["by_split_and_behavior"][split]["all"]
        print(f"{split}: n={row.get('examples',0)} target_metadata={row['rates']['target_in_metadata_rate']:.4f} "
              f"target_train_vocab={row['rates']['target_in_train_vocab_rate']:.4f} "
              f"target_train_click={row['rates']['target_in_train_targets_rate']:.4f}")


if __name__ == "__main__":
    main()
