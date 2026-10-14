"""Post-hoc behavioral evaluation of a saved v0.4 attention-GRU checkpoint.

Evaluation only: no training, model selection, or permutation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from travelerreclab.models.gru_attention import create_model
from travelerreclab.models.gru_diagnostics import (
    PATTERNS, expected_ranking_metrics, pattern_for,
)
from travelerreclab.models.mean_pool import iter_batches, make_batch


def analyze(model, path, vocabulary, *, split='validation', batch_size=256,
            max_examples=None, device='cpu'):
    import torch

    model.eval()
    sums = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    with torch.no_grad():
        for rows in iter_batches(path, split, batch_size, max_examples=max_examples):
            history, candidates, targets = make_batch(rows, vocabulary, device)
            metrics = expected_ranking_metrics(model(history, candidates), targets)
            for row, values in zip(rows, metrics):
                group = pattern_for(row)
                for key in ('all', group):
                    sums[key][0] += 1
                    for i, value in enumerate(values, 1):
                        sums[key][i] += value
    return {
        key: {'n': vals[0],
              'mrr_neutral_ties': vals[1] / vals[0],
              'recall@5_neutral_ties': vals[2] / vals[0],
              'ndcg@5_neutral_ties': vals[3] / vals[0]}
        for key, vals in sums.items() if vals[0]
    }


def run(checkpoint, path, output, *, split='validation', batch_size=256,
        max_examples=None, device='cpu'):
    import torch

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('architecture') != 'hotel_id_candidate_attention_gru':
        raise ValueError('Expected hotel_id_candidate_attention_gru checkpoint')
    vocabulary = saved['vocabulary']
    model = create_model(max(vocabulary.values(), default=1) + 1, int(saved['dim']))
    model.load_state_dict(saved['model'])
    model.to(device)
    groups = analyze(model, path, vocabulary, split=split, batch_size=batch_size,
                     max_examples=max_examples, device=device)
    report = {
        'split': split,
        'checkpoint': str(checkpoint),
        'vocabulary_size': len(vocabulary),
        'max_examples': max_examples,
        'metrics': groups,
        'notes': [
            'Same original retained-history behavioral taxonomy as v0.3.1.',
            'No shuffled inputs, retraining, checkpoint selection, or test tuning.',
            'Metrics are post-hoc descriptive and examples within sessions are correlated.',
            'Candidate lists are reconstructed from target clickout rows.',
        ],
    }
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', default='data/processed/next_clickout.jsonl')
    parser.add_argument('--output', default='data/processed/gru_attention_v04_diagnostics.json')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-examples', type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    report = run(args.checkpoint, args.input, args.output, split=args.split,
                 batch_size=args.batch_size, max_examples=args.max_examples,
                 device=args.device)
    print('Report:', args.output)
    for name in ('all', *PATTERNS):
        if name in report['metrics']:
            print(name, report['metrics'][name])


if __name__ == '__main__':
    main()
