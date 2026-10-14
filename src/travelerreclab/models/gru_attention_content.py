"""v0.5.1: candidate-aware GRU with ID + property-tag hotel encodings.

Content tags are not guaranteed point-in-time correct in the source snapshot.
Candidate lists remain reconstructed from target click-out rows.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

from travelerreclab.models.mean_pool import (
    build_vocabulary, encode, hotel_history, iter_batches,
)
from travelerreclab.models.gru_attention import create_model as create_id_model


def load_properties(path: str | Path):
    """Load exact raw hotel IDs; tag vocabulary is source metadata, never targets.

    Tags are attributes, not outcome labels, but historical availability is unverified.
    """
    properties = {}
    tags = set()
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'item_id', 'properties'}.issubset(reader.fieldnames or ()):
            raise ValueError('Metadata needs item_id and properties columns')
        for row in reader:
            hotel = (row.get('item_id') or '').strip()
            if not hotel or hotel in properties:
                continue
            names = tuple(sorted({x.strip() for x in (row.get('properties') or '').split('|') if x.strip()}))
            properties[hotel] = names
            tags.update(names)
    tag_vocab = {name: i + 1 for i, name in enumerate(sorted(tags))}
    encoded = {hotel: tuple(tag_vocab[tag] for tag in names) for hotel, names in properties.items()}
    return encoded, tag_vocab


def make_content_batch(rows, vocab, props, device='cpu'):
    import torch
    encoded = [encode(e, vocab) for e in rows]
    raw_hist = [hotel_history(e) for e in rows]
    raw_cand = [[str(x) for x in e['candidate_items']] for e in rows]
    mh = max(1, *(len(x[0]) for x in encoded))
    mc = max(len(x[1]) for x in encoded)
    hp = [[props.get(h, ()) for h in history] for history in raw_hist]
    cp = [[props.get(h, ()) for h in candidates] for candidates in raw_cand]
    max_props = max(1, *(len(v) for example in hp + cp for v in example))
    hist = torch.zeros((len(rows), mh), dtype=torch.long, device=device)
    cand = torch.zeros((len(rows), mc), dtype=torch.long, device=device)
    htags = torch.zeros((len(rows), mh, max_props), dtype=torch.long, device=device)
    ctags = torch.zeros((len(rows), mc, max_props), dtype=torch.long, device=device)
    target = torch.tensor([x[2] for x in encoded], dtype=torch.long, device=device)
    for i, (h, c, _) in enumerate(encoded):
        if h:
            hist[i, :len(h)] = torch.tensor(h, device=device)
        cand[i, :len(c)] = torch.tensor(c, device=device)
        for j, tags in enumerate(hp[i]):
            if tags:
                htags[i, j, :len(tags)] = torch.tensor(tags, device=device)
        for j, tags in enumerate(cp[i]):
            if tags:
                ctags[i, j, :len(tags)] = torch.tensor(tags, device=device)
    return hist, cand, htags, ctags, target


def create_model(vocab_size: int, n_tags: int, dim: int, content_weight: float = 0.5):
    """Keep exact v0.4 GRU/attention structure, add mean property vectors.

    Hotel-ID embedding/bias of unseen candidates still share UNK; their
    content vectors can differ. Zero-tag hotels have zero content contribution.
    """
    import torch
    from torch import nn
    if not 0 <= content_weight <= 1:
        raise ValueError('content_weight must be in [0,1]')

    class ContentAttentionGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.hotels = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.bias = nn.Embedding(vocab_size, 1, padding_idx=0)
            self.tags = nn.Embedding(n_tags + 1, dim, padding_idx=0)
            self.gru = nn.GRU(dim, dim, batch_first=True)
            self.query = nn.Linear(dim, dim, bias=False)
            # Content-only prior is essential when no hotel history exists.
            self.content_prior = nn.Linear(dim, 1, bias=False)
            self.mix_logit = nn.Parameter(torch.tensor(0.0))
            self.empty = nn.Parameter(torch.zeros(dim))
            self.scale = dim ** -0.5
            self.content_weight = content_weight

        def property_vector(self, prop_ids):
            mask = prop_ids.ne(0)
            counts = mask.sum(dim=-1).clamp(min=1).unsqueeze(-1)
            content = self.tags(prop_ids).sum(dim=-2) / counts
            return content

        def hotel_vector(self, ids, prop_ids):
            return self.hotels(ids) + self.content_weight * self.property_vector(prop_ids)

        def forward(self, hist, cand, htags, ctags):
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            lengths = hist.ne(0).sum(dim=1)
            nonempty = lengths > 0
            batch_size = cand.size(0)
            hotel_vectors = self.hotel_vector(cand, ctags)
            traveler = self.empty.unsqueeze(0).expand(batch_size, -1).clone()
            context = torch.zeros_like(hotel_vectors)
            if nonempty.any():
                valid_lengths = lengths[nonempty]
                history_vectors = self.hotel_vector(hist[nonempty], htags[nonempty])
                packed = pack_padded_sequence(history_vectors, valid_lengths.cpu(),
                                              batch_first=True, enforce_sorted=False)
                packed_outputs, last = self.gru(packed)
                states, _ = pad_packed_sequence(packed_outputs, batch_first=True,
                                                total_length=hist.size(1))
                traveler[nonempty] = last[-1]
                logits = torch.bmm(self.query(hotel_vectors[nonempty]),
                                   states.transpose(1, 2)) * self.scale
                position_mask = (torch.arange(hist.size(1), device=hist.device)[None, :]
                                 < valid_lengths[:, None])
                weights = torch.softmax(logits.masked_fill(~position_mask[:, None, :], float('-inf')), dim=-1)
                context[nonempty] = torch.bmm(weights, states)
            scores = (hotel_vectors * (traveler[:, None, :] + torch.sigmoid(self.mix_logit) * context)).sum(-1) * self.scale
            scores += self.bias(cand).squeeze(-1)
            scores += self.content_prior(self.property_vector(ctags)).squeeze(-1)
            return scores.masked_fill(cand.eq(0), float('-inf'))

    return ContentAttentionGRU()


def evaluate_content(model, path, split, vocab, props, batch_size, max_examples=None, device='cpu'):
    import torch
    import math
    model.eval()
    n = 0
    reciprocal = recall5 = ndcg5 = 0.
    unknown_target = empty_hist = no_target_props = 0
    with torch.no_grad():
        for rows in iter_batches(path, split, batch_size, max_examples=max_examples):
            h, c, ht, ct, targets = make_content_batch(rows, vocab, props, device)
            scores = model(h, c, ht, ct)
            gold = scores.gather(1, targets[:, None])
            better = (scores > gold).sum(1).tolist()
            tied = (scores == gold).sum(1).tolist()
            for b, t in zip(better, tied):
                positions = range(b + 1, b + t + 1)
                reciprocal += sum(1/r for r in positions)/t
                recall5 += sum(r <= 5 for r in positions)/t
                ndcg5 += sum((1/math.log2(r+1)) if r <= 5 else 0 for r in positions)/t
            unknown_target += sum(str(e['target_item']) not in vocab for e in rows)
            no_target_props += sum(not props.get(str(e['target_item'])) for e in rows)
            empty_hist += sum(not hotel_history(e) for e in rows)
            n += len(rows)
    return {'n': n, 'mrr_neutral_ties': reciprocal/n if n else None,
            'recall@5_neutral_ties': recall5/n if n else None,
            'ndcg@5_neutral_ties': ndcg5/n if n else None,
            'unknown_target_rate': unknown_target/n if n else None,
            'empty_hotel_history_rate': empty_hist/n if n else None,
            'target_missing_properties_rate': no_target_props/n if n else None}


def train(path, metadata, output, *, dim=32, epochs=3, batch_size=256, lr=.003,
          max_train=None, max_eval=None, min_count=1, seed=42, device='cpu', content_weight=0.5):
    import torch
    from torch.nn import functional as F
    if min(dim, epochs, batch_size) < 1 or lr <= 0:
        raise ValueError('Invalid hyperparameters')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    vocab = build_vocabulary(path, min_count=min_count, max_train=max_train)
    props, tag_vocab = load_properties(metadata)
    model = create_model(max(vocab.values(), default=1) + 1, len(tag_vocab), dim, content_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    best, best_state = float('-inf'), None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, seen = 0., 0
        for rows in iter_batches(path, 'train', batch_size, max_examples=max_train):
            h, c, ht, ct, t = make_content_batch(rows, vocab, props, device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(h, c, ht, ct), t)
            loss.backward()
            opt.step()
            loss_total += loss.item() * len(rows)
            seen += len(rows)
        validation = evaluate_content(model, path, 'validation', vocab, props, batch_size, max_eval, device)
        trace.append({'epoch': epoch, 'train_n': seen,
                      'train_mean_ce': loss_total / seen if seen else None, 'validation': validation})
        score = validation['mrr_neutral_ties']
        if score is not None and score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'epoch={epoch} train_n={seen} loss={trace[-1]["train_mean_ce"]} val_MRR={score}', flush=True)
    if best_state is None:
        raise ValueError('No validation examples to select a checkpoint')
    model.load_state_dict(best_state)
    test = evaluate_content(model, path, 'test', vocab, props, batch_size, max_eval, device)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix('.pt')
    torch.save({'model': best_state, 'vocabulary': vocab, 'tag_vocabulary': tag_vocab,
                'dim': dim, 'content_weight': content_weight, 'architecture': 'id_plus_property_attention_gru'}, ckpt)
    report = {'configuration': {'dim': dim, 'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
             'min_count': min_count, 'seed': seed, 'max_train': max_train,
             'max_eval_per_split': max_eval, 'device': device, 'content_weight': content_weight,
             'metadata': str(metadata)},
             'vocab_size': len(vocab), 'property_vocab_size': len(tag_vocab),
             'training': trace, 'best_epoch': max(range(len(trace)), key=lambda i: trace[i]['validation']['mrr_neutral_ties'])+1,
             'selection': 'maximum validation tie-neutral MRR; first epoch on ties',
             'test': test, 'checkpoint': str(ckpt),
             'limitations': [
                 'Metadata properties are a snapshot without verified point-in-time availability.',
                 'Candidate lists come from target clickout rows, not verified pre-click exposures.',
                 'Raw IDs absent from training share ID UNK and bias, but can differ via available property tags.',
                 'Property vocabulary derived from all metadata, including hotel records not seen in training; no click labels are used.',
                 'History limited to retained preceding hotel-bearing actions; no event context, time gaps or candidate position.',
                 'Additional content features and parameters mean this is not parameter-count matched to v0.4.',
                 'File-order training; initial 3 epochs do not prove convergence.',
                 'No test-driven tuning or booking/conversion claims.']}
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/processed/next_clickout.jsonl')
    p.add_argument('--metadata', default='data/raw/item_metadata.csv')
    p.add_argument('--output', default='data/processed/gru_attention_content_v051.json')
    p.add_argument('--dim', type=int, default=32)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--content-weight', type=float, default=0.5)
    p.add_argument('--max-train', type=int)
    p.add_argument('--max-eval', type=int)
    p.add_argument('--min-count', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    result = train(a.input, a.metadata, a.output, dim=a.dim, epochs=a.epochs,
                   batch_size=a.batch_size, lr=a.lr, content_weight=a.content_weight,
                   max_train=a.max_train, max_eval=a.max_eval, min_count=a.min_count,
                   seed=a.seed, device=a.device)
    print('Report:', a.output, 'Test:', result['test'])


if __name__ == '__main__':
    main()
