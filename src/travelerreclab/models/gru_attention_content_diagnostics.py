"""Evaluation-only v0.5.1 ranking slices using the validation-selected checkpoint.

The target-row candidate set is reconstructed; metadata is not point-in-time verified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from travelerreclab.data.audit_hotel_metadata_v05 import behavior
from travelerreclab.models.mean_pool import hotel_history, iter_batches
from travelerreclab.models.gru_attention_content import (
    create_model, load_properties, make_content_batch,
)

METRICS = ('mrr_neutral_ties', 'recall@5_neutral_ties', 'ndcg@5_neutral_ties')
BEHAVIORS = ('immediate_continuation', 'return_to_earlier_hotel',
             'new_hotel_after_exploration', 'no_prior_hotel_reference')


def train_target_ids(path):
    """Positive training labels ONLY, not candidate exposure."""
    from travelerreclab.evaluation.baselines import read_examples, split_of, valid_example
    return {str(e['target_item']) for e in read_examples(path)
            if valid_example(e) and split_of(e['target_timestamp']) == 'train'}


def target_coverage(target, vocab, positives):
    if target not in vocab:
        return 'outside_train_vocabulary'
    return 'clicked_in_train' if target in positives else 'in_vocab_not_clicked_in_train'


def reciprocal_metrics(scores, target_index):
    gold = scores[target_index]
    better = sum(s > gold for s in scores)
    tied = sum(s == gold for s in scores)
    if tied < 1:
        raise AssertionError('Target score not found among candidates')
    ranks = range(better + 1, better + tied + 1)
    return (
        sum(1 / r for r in ranks) / tied,
        sum(r <= 5 for r in ranks) / tied,
        sum((1 / math.log2(r + 1)) if r <= 5 else 0 for r in ranks) / tied,
    )


def report_slices(checkpoint, path, metadata, split, *, batch_size=256,
                  max_examples=None, device='cpu'):
    import torch
    if split not in ('validation', 'test'):
        raise ValueError('Only validation and test are supported')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('architecture') != 'id_plus_property_attention_gru':
        raise ValueError('Unexpected checkpoint architecture')
    vocab = saved['vocabulary']
    props, tags = load_properties(metadata)
    if tags != saved['tag_vocabulary']:
        raise ValueError('Metadata property vocabulary does not match checkpoint')
    positives = train_target_ids(path)
    model = create_model(max(vocab.values(), default=1) + 1,
                         len(tags), saved['dim'], saved['content_weight']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    aggregates = defaultdict(lambda: [0, 0., 0., 0.])
    with torch.no_grad():
        for rows in iter_batches(path, split, batch_size, max_examples=max_examples):
            h, c, ht, ct, targets = make_content_batch(rows, vocab, props, device)
            scores = model(h, c, ht, ct).cpu().tolist()
            for e, row_scores, idx in zip(rows, scores, targets.tolist()):
                target = str(e['target_item'])
                hist = hotel_history(e)
                group = behavior(e, hist)
                coverage = target_coverage(target, vocab, positives)
                metadata_state = 'with_properties' if props.get(target) else 'without_properties'
                vals = reciprocal_metrics(row_scores[:len(e['candidate_items'])], idx)
                for label in ('all', f'behavior/{group}', f'train_target/{coverage}',
                              f'properties/{metadata_state}',
                              f'behavior_and_train_target/{group}/{coverage}'):
                    agg = aggregates[label]
                    agg[0] += 1
                    for j, v in enumerate(vals, 1):
                        agg[j] += v
    result = {label: {'n': agg[0], **{metric: agg[i + 1] / agg[0]
                          for i, metric in enumerate(METRICS)}}
              for label, agg in sorted(aggregates.items()) if agg[0]}
    return {'split': split, 'checkpoint': str(checkpoint), 'metadata': str(metadata),
            'max_examples': max_examples, 'metrics': result,
            'notes': [
                'Train vocabulary is read from the saved checkpoint (history and train candidates).',
                'Train target membership is computed from positive training clickouts.',
                'Behavior patterns use retained recognized hotel history.',
                'All comparisons are post hoc and are not a basis for test-driven tuning.',
                'Unseen target ID slices are small; unknown candidates may be present in other slices.',
                'No session-clustered confidence intervals; examples are correlated.',
                'Candidate sets are target-row reconstructions; metadata is not point-in-time verified.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='data/processed/gru_attention_content_v051.pt')
    parser.add_argument('--input', default='data/processed/next_clickout.jsonl')
    parser.add_argument('--metadata', default='data/raw/item_metadata.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-examples', type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    result = report_slices(args.checkpoint, args.input, args.metadata, args.split,
                           batch_size=args.batch_size, max_examples=args.max_examples,
                           device=args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Report:', path)
    for label in ('all', *(f'behavior/{name}' for name in BEHAVIORS)):
        print(label, result['metrics'].get(label))


if __name__ == '__main__':
    main()
