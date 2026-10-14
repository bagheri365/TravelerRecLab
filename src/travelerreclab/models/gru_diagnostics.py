"""Evaluation-only GRU order perturbation and behavioral-slice audit.

Loads v0.3.1 checkpoint; never trains or selects new model parameters.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from travelerreclab.models.gru import create_model
from travelerreclab.models.mean_pool import hotel_history, iter_batches, make_batch

PATTERNS = ('immediate_continuation', 'return_to_earlier_hotel',
            'new_hotel_after_exploration', 'no_prior_hotel_reference')


def pattern_for(row):
    hotels = hotel_history(row)
    target = str(row['target_item'])
    if not hotels:
        return 'no_prior_hotel_reference'
    if hotels[-1] == target:
        return 'immediate_continuation'
    if target in hotels:
        return 'return_to_earlier_hotel'
    return 'new_hotel_after_exploration'


def shuffled_histories(hist, *, seed, offsets):
    """Permute only nonpad tokens independently per example and fixed global index."""
    altered = hist.clone()
    for i, global_index in enumerate(offsets):
        length = int(hist[i].ne(0).sum().item())
        if length < 2:
            continue
        # Stable across batch boundaries, Python processes, and platforms.
        digest = hashlib.sha256(f'{seed}:{global_index}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        order = list(range(length))
        rng.shuffle(order)
        altered[i, :length] = hist[i, order]
    return altered


def expected_ranking_metrics(scores, targets):
    """Tie-neutral RR, Recall@5 and NDCG@5 per row."""
    gold = scores.gather(1, targets[:, None])
    better = (scores > gold).sum(dim=1).tolist()
    tied = (scores == gold).sum(dim=1).tolist()
    output = []
    for b, n in zip(better, tied):
        ranks = range(b + 1, b + n + 1)
        output.append((sum(1 / r for r in ranks) / n,
                       sum(r <= 5 for r in ranks) / n,
                       sum(1 / math.log2(r + 1) for r in ranks if r <= 5) / n))
    return output


def analyze(model, path, vocab, *, split='test', batch_size=256,
            max_examples=None, seed=42, device='cpu'):
    import torch
    model.eval()
    sums = {mode: defaultdict(lambda: [0, 0., 0., 0.])
            for mode in ('original', 'shuffled')}
    changed = 0
    eligible = 0
    count = 0
    with torch.no_grad():
        for rows in iter_batches(path, split, batch_size, max_examples=max_examples):
            h, c, t = make_batch(rows, vocab, device)
            hs = shuffled_histories(h, seed=seed, offsets=range(count, count + len(rows)))
            changed += int((hs != h).any(dim=1).sum().item())
            eligible += int(h.ne(0).sum(dim=1).ge(2).sum().item())
            patterns = [pattern_for(row) for row in rows]
            for mode, history in (('original', h), ('shuffled', hs)):
                results = expected_ranking_metrics(model(history, c), t)
                for group, metrics in zip(patterns, results):
                    for label in ('all', group):
                        a = sums[mode][label]
                        a[0] += 1
                        for k, value in enumerate(metrics, start=1):
                            a[k] += value
            count += len(rows)
    def finalize(groups):
        return {group: {'n': a[0], 'mrr_neutral_ties': a[1]/a[0],
                        'recall@5_neutral_ties': a[2]/a[0],
                        'ndcg@5_neutral_ties': a[3]/a[0]}
                for group, a in groups.items() if a[0]}
    originals, shuffles = (finalize(sums[x]) for x in ('original', 'shuffled'))
    return {'split': split, 'seed': seed, 'max_examples': max_examples,
            'original': originals, 'shuffled': shuffles,
            'permutation': {'eligible_length_ge_2': eligible,
                            'actually_changed_order': changed,
                            'unchanged_after_shuffle': count-changed},
            'notes': [
                'Order perturbation is evaluation-only: fixed checkpoint and hotel-ID multisets.',
                'The RNG depends on global example index; batch size does not affect permutation.',
                'History length 0/1 and permutations that happen to preserve ID order are unchanged.',
                'A drop under shuffled order measures sensitivity, not causal attribution to sequence order.',
                'Test slice breakdown is post hoc diagnostic, not model/parameter selection.',
                'Multiple clickouts per session make examples correlated; no clustered uncertainty estimates.'
            ]}


def run(checkpoint, path, output, *, split='test', seed=42, batch_size=256,
        max_examples=None, device='cpu'):
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('architecture') != 'hotel_id_gru':
        raise ValueError('Expected hotel_id_gru checkpoint')
    vocab = saved['vocabulary']
    dim = int(saved['dim'])
    model = create_model(max(vocab.values(), default=1)+1, dim)
    model.load_state_dict(saved['model'])
    model.to(device)
    report = analyze(model, path, vocab, split=split, batch_size=batch_size,
                     max_examples=max_examples, seed=seed, device=device)
    report['checkpoint'] = str(checkpoint)
    report['vocabulary_size'] = len(vocab)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', default='data/processed/next_clickout.jsonl')
    p.add_argument('--output', default='data/processed/gru_v031_diagnostics.json')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-examples', type=int)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    report = run(a.checkpoint, a.input, a.output, split=a.split, seed=a.seed,
                 batch_size=a.batch_size, max_examples=a.max_examples, device=a.device)
    print('Report:', a.output)
    for mode in ('original', 'shuffled'):
        print(mode, report[mode]['all'])
    print('Shuffled examples changed:', report['permutation']['actually_changed_order'])


if __name__ == '__main__':
    main()
