"""First learned traveler encoder: train-only hotel-ID embeddings + mean pooling.

Candidate-set softmax is an offline reconstructed click-out benchmark, NOT an
independently timestamped pre-click exposure log. No candidate position input.
Requires optional PyTorch (`pip install torch`).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from travelerreclab.evaluation.baselines import HOTEL_ACTIONS, read_examples, split_of, valid_example


def hotel_history(e: dict) -> list[str]:
    """Do not mistake destinations/filters for hotel IDs."""
    return [str(h['reference']) for h in e.get('history', [])
            if h.get('action_type') in HOTEL_ACTIONS and h.get('reference') not in (None, '')]


def build_vocabulary(path: str | Path, *, min_count: int = 1, max_train: int | None = None) -> dict[str, int]:
    if min_count < 1:
        raise ValueError('min_count must be positive')
    counts: Counter = Counter()
    seen = 0
    for e in read_examples(path):
        if split_of(e['target_timestamp']) != 'train' or not valid_example(e):
            continue
        counts.update(hotel_history(e))
        counts.update(map(str, e['candidate_items']))
        seen += 1
        if max_train is not None and seen >= max_train:
            break
    # 0 = padding/no history, 1 = unknown hotel; never fit on validation/test IDs.
    return {item: i + 2 for i, (item, n) in enumerate(sorted(counts.items())) if n >= min_count}


def encode(e: dict, vocab: dict[str, int]) -> tuple[list[int], list[int], int]:
    if not valid_example(e):
        raise ValueError('Invalid example / target outside candidates')
    history = [vocab.get(h, 1) for h in hotel_history(e)]
    candidates = [vocab.get(str(h), 1) for h in e['candidate_items']]
    target = [str(x) for x in e['candidate_items']].index(str(e['target_item']))
    return history, candidates, target


def make_batch(rows: list[dict], vocab: dict[str, int], device: str = 'cpu'):
    import torch
    encoded = [encode(e, vocab) for e in rows]
    mh = max(1, *(len(x[0]) for x in encoded))
    mc = max(len(x[1]) for x in encoded)
    hist = torch.zeros((len(rows), mh), dtype=torch.long, device=device)
    cand = torch.zeros((len(rows), mc), dtype=torch.long, device=device)
    target = torch.tensor([x[2] for x in encoded], dtype=torch.long, device=device)
    for i, (h, c, _) in enumerate(encoded):
        if h:
            hist[i, :len(h)] = torch.tensor(h, device=device)
        cand[i, :len(c)] = torch.tensor(c, device=device)
    return hist, cand, target


def create_model(vocab_size: int, dim: int):
    import torch
    from torch import nn

    class MeanPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.hotels = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.bias = nn.Embedding(vocab_size, 1, padding_idx=0)
            self.empty = nn.Parameter(torch.zeros(dim))
            self.scale = dim ** -0.5

        def forward(self, hist, cand):
            mask = (hist != 0).unsqueeze(-1)
            total = (self.hotels(hist) * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            traveler = total / count
            traveler = traveler + (mask.sum(dim=1) == 0).to(traveler.dtype) * self.empty
            scores = (self.hotels(cand) * traveler.unsqueeze(1)).sum(dim=-1) * self.scale
            scores = scores + self.bias(cand).squeeze(-1)
            return scores.masked_fill(cand == 0, float('-inf'))

    return MeanPool()


def iter_batches(path, split, batch_size, *, max_examples=None):
    batch = []
    n = 0
    for e in read_examples(path):
        if split_of(e['target_timestamp']) != split or not valid_example(e):
            continue
        if max_examples is not None and n >= max_examples:
            break
        batch.append(e)
        n += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def evaluate(model, path, split, vocab, batch_size, max_examples=None, device='cpu'):
    import torch
    model.eval()
    n, reciprocal, hit5, ndcg5, unknown_target, empty_hist = 0, 0., 0., 0., 0, 0
    with torch.no_grad():
        for rows in iter_batches(path, split, batch_size, max_examples=max_examples):
            h, c, t = make_batch(rows, vocab, device)
            scores = model(h, c)
            # Expected reciprocal rank under uniform random tie ordering, not logged-order tie bias.
            gold = scores.gather(1, t[:, None])
            better = (scores > gold).sum(dim=1)
            tied = (scores == gold).sum(dim=1)
            for b, tie in zip(better.tolist(), tied.tolist()):
                ranks = range(b + 1, b + tie + 1)
                reciprocal += sum(1 / r for r in ranks) / tie
                hit5 += sum(r <= 5 for r in ranks) / tie
                import math
                ndcg5 += sum(1 / math.log2(r + 1) if r <= 5 else 0 for r in ranks) / tie
            unknown_target += sum(str(e['target_item']) not in vocab for e in rows)
            empty_hist += sum(not hotel_history(e) for e in rows)
            n += len(rows)
    return {'n': n, 'mrr_neutral_ties': reciprocal/n if n else None,
            'recall@5_neutral_ties': hit5/n if n else None,
            'ndcg@5_neutral_ties': ndcg5/n if n else None,
            'unknown_target_rate': unknown_target/n if n else None,
            'empty_hotel_history_rate': empty_hist/n if n else None}


def train(path, output, *, dim=32, epochs=3, batch_size=256, lr=.003,
          max_train=None, max_eval=None, min_count=1, seed=42, device='cpu'):
    import torch
    from torch.nn import functional as F
    if min(dim, epochs, batch_size) < 1 or lr <= 0:
        raise ValueError('Invalid training hyperparameters')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    vocab = build_vocabulary(path, min_count=min_count, max_train=max_train)
    model = create_model(max(vocab.values(), default=1) + 1, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    best = float('-inf')
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.
        seen = 0
        for rows in iter_batches(path, 'train', batch_size, max_examples=max_train):
            h, c, t = make_batch(rows, vocab, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(h, c), t)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(rows)
            seen += len(rows)
        validation = evaluate(model, path, 'validation', vocab, batch_size, max_eval, device)
        trace.append({'epoch': epoch, 'train_n': seen,
                      'train_mean_ce': loss_total / seen if seen else None, 'validation': validation})
        score = validation['mrr_neutral_ties']
        if score is not None and score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch} train_n={seen} loss={trace[-1]['train_mean_ce']} val_MRR={score}", flush=True)
    if best_state is None:
        raise ValueError('No validation examples; cannot select a checkpoint')
    model.load_state_dict(best_state)
    # One test evaluation AFTER selecting checkpoint on validation only.
    test = evaluate(model, path, 'test', vocab, batch_size, max_eval, device)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix('.pt')
    torch.save({'model': best_state, 'vocabulary': vocab, 'dim': dim, 'seed': seed}, ckpt)
    report = {'configuration': {'dim': dim, 'epochs': epochs, 'batch_size': batch_size,
              'lr': lr, 'min_count': min_count, 'seed': seed, 'max_train': max_train,
              'max_eval_per_split': max_eval, 'device': device},
              'vocab_size': len(vocab), 'training': trace,
              'selection': 'maximum validation tie-neutral MRR; first epoch on ties',
              'best_epoch': max(range(len(trace)), key=lambda i: trace[i]['validation']['mrr_neutral_ties'])+1,
              'test': test, 'checkpoint': str(ckpt),
              'limitations': ['Item-ID-only model cannot generalize semantically to unseen hotels; IDs not seen in training map to UNK.',
              'Unseen history and candidates share the same UNK token; report unknown-target rate.',
              'Candidate lists come from clicked-out target rows; not separately proven pre-click exposures.',
              'History consists of retained preceding events only (maximum 50); zero-history clickouts excluded.',
              'No candidate position, price, metadata, or search-context features; comparison to hybrid is not feature matched.',
              'Training order follows JSONL file order, without buffered shuffling; initial reference implementation.',
              'Do not use test results to choose architecture, hyperparameters, or epoch.']}
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/processed/next_clickout.jsonl')
    p.add_argument('--output', default='data/processed/mean_pool_v03.json')
    p.add_argument('--dim', type=int, default=32)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--max-train', type=int)
    p.add_argument('--max-eval', type=int)
    p.add_argument('--min-count', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    result = train(args.input, args.output, dim=args.dim, epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.lr, max_train=args.max_train,
                   max_eval=args.max_eval, min_count=args.min_count, seed=args.seed,
                   device=args.device)
    print('Report:', args.output, 'Test:', result['test'])


if __name__ == '__main__':
    main()
