"""Candidate-aware GRU attention over earlier hotel interactions (v0.4).

The same train-only vocabulary, batches, cross-entropy objective, and
tie-neutral evaluation as v0.3.1. No candidate position or target-row features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from travelerreclab.models.mean_pool import (
    build_vocabulary, evaluate, iter_batches, make_batch,
)


def create_model(vocab_size: int, dim: int):
    """Candidate-conditioned attention over all GRU states, with final-state skip.

    The skip connection keeps a direct continuation path while attention allows
    retrieval from earlier states. Attention never observes target labels.
    """
    import torch
    from torch import nn

    class CandidateAwareGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.hotels = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.bias = nn.Embedding(vocab_size, 1, padding_idx=0)
            self.gru = nn.GRU(dim, dim, batch_first=True)
            self.query = nn.Linear(dim, dim, bias=False)
            # A shared (not label-dependent) blend of final GRU state and
            # candidate-specific attention context. Sigmoid ensures [0,1].
            self.mix_logit = nn.Parameter(torch.tensor(0.0))
            self.empty = nn.Parameter(torch.zeros(dim))
            self.scale = dim ** -0.5

        def forward(self, hist, cand):
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

            lengths = hist.ne(0).sum(dim=1)
            nonempty = lengths > 0
            batch_size, n_candidates = cand.shape
            hotel_vectors = self.hotels(cand)
            traveler = self.empty.unsqueeze(0).expand(batch_size, -1).clone()
            # Empty histories bypass softmax over empty positions entirely.
            context = torch.zeros_like(hotel_vectors)
            if nonempty.any():
                valid_h = hist[nonempty]
                valid_lengths = lengths[nonempty]
                packed = pack_padded_sequence(self.hotels(valid_h),
                    valid_lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, final_state = self.gru(packed)
                states, _ = pad_packed_sequence(packed_output, batch_first=True,
                                                total_length=hist.size(1))
                traveler[nonempty] = final_state[-1]
                # Queries depend on candidate identity; keys/values are GRU states.
                queries = self.query(hotel_vectors[nonempty])
                logits = torch.bmm(queries, states.transpose(1, 2)) * self.scale
                valid_positions = torch.arange(hist.size(1), device=hist.device)[None, :] < valid_lengths[:, None]
                logits = logits.masked_fill(~valid_positions[:, None, :], float('-inf'))
                attention = torch.softmax(logits, dim=-1)
                context[nonempty] = torch.bmm(attention, states)
            blend = torch.sigmoid(self.mix_logit)
            representation = traveler.unsqueeze(1) + blend * context
            scores = (hotel_vectors * representation).sum(dim=-1) * self.scale
            scores = scores + self.bias(cand).squeeze(-1)
            return scores.masked_fill(cand == 0, float('-inf'))

    return CandidateAwareGRU()


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
        loss_total, seen = 0., 0
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
                      'train_mean_ce': loss_total / seen if seen else None,
                      'validation': validation})
        score = validation['mrr_neutral_ties']
        if score is not None and score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'epoch={epoch} train_n={seen} loss={trace[-1]["train_mean_ce"]} val_MRR={score}', flush=True)
    if best_state is None:
        raise ValueError('No validation examples; cannot select checkpoint')
    model.load_state_dict(best_state)
    test = evaluate(model, path, 'test', vocab, batch_size, max_eval, device)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix('.pt')
    torch.save({'model': best_state, 'vocabulary': vocab, 'dim': dim, 'seed': seed,
                'architecture': 'hotel_id_candidate_attention_gru'}, ckpt)
    report = {
        'configuration': {'dim': dim, 'epochs': epochs, 'batch_size': batch_size,
                          'lr': lr, 'min_count': min_count, 'seed': seed,
                          'max_train': max_train, 'max_eval_per_split': max_eval, 'device': device},
        'vocab_size': len(vocab), 'training': trace,
        'selection': 'maximum validation tie-neutral MRR; first epoch on ties',
        'best_epoch': max(range(len(trace)), key=lambda i: trace[i]['validation']['mrr_neutral_ties']) + 1,
        'test': test, 'checkpoint': str(ckpt),
        'limitations': [
            'Hotel-ID-only inputs: no action-type, time-gap, price, hotel metadata, candidate position, or search context.',
            'Hotel IDs absent from train vocabulary share UNK embedding; cannot distinguish cold-start hotels.',
            'History uses retained preceding events only (maximum 50); zero-history clickouts excluded.',
            'Candidate lists come from target clickout rows, not independently verified pre-click exposures.',
            'Training streams examples in file order without shuffled mini-batches.',
            'Compared with v0.3.1, candidate-conditioned attention and a final-state skip are added; parameter counts differ.',
            'Do not use test results to choose hyperparameters, epochs, or architectures.'
        ]
    }
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/processed/next_clickout.jsonl')
    parser.add_argument('--output', default='data/processed/gru_attention_v04.json')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--max-train', type=int)
    parser.add_argument('--max-eval', type=int)
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    result = train(args.input, args.output, dim=args.dim, epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.lr,
                   max_train=args.max_train, max_eval=args.max_eval,
                   min_count=args.min_count, seed=args.seed, device=args.device)
    print('Report:', args.output, 'Test:', result['test'])


if __name__ == '__main__':
    main()
