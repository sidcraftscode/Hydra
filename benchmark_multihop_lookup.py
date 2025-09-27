"""Multi-hop lookup benchmark (2–6 hops).

Data: synthetic KB chains E1->E2, E2->E3, ... with distractors.
Format: Facts list + <scratch> steps + query; model predicts final entity.
Runs: Hydra (PKM ON vs PKM OFF) vs Transformer.
Output: Accuracy vs hops (CSV/JSON in results/).
"""
from __future__ import annotations
import os, math, random, json, csv, time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F

from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters


# ------------------ Synthetic tokenizer ------------------
@dataclass
class Vocab:
    n_entities: int
    arrow: int
    semi: int
    q: int
    scratch: int
    eq: int
    @property
    def size(self):
        return self.eq + 1

def make_vocab(n_entities: int = 512) -> Vocab:
    return Vocab(n_entities=n_entities,
                 arrow=n_entities,
                 semi=n_entities+1,
                 q=n_entities+2,
                 scratch=n_entities+3,
                 eq=n_entities+4)


# ------------------ Data generation ------------------
def gen_chain_sequence(start: int, hops: int, vocab: Vocab, max_entity: int, distractors: int = 160,
                       scratch_fraction: float = 0.7) -> Tuple[List[int], int]:
    """Create one sequence encoding a chain window with distractor facts and a final query.

    Returns tokens list and target entity id for final answer.
    """
    # Ground-truth chain with random next hops unique to this instance
    chain_edges: List[Tuple[int,int]] = []
    cur = start
    path = [cur]
    used_nodes = {cur}
    for _ in range(hops):
        # sample next distinct node not in path to avoid trivial arithmetic rule
        # try up to few times
        for _try in range(8):
            nxt = random.randrange(0, max_entity)
            if nxt not in used_nodes:
                break
    used_nodes.add(nxt)
    chain_edges.append((cur, nxt))
    path.append(nxt)
    cur = nxt
    target = cur

    # Add distractor edges sampled far from the path
    used = set(path)
    facts: List[Tuple[int,int]] = chain_edges.copy()
    for _ in range(distractors):
        a = random.randrange(0, max_entity)
        # try sampling away from path to avoid trivial cues
        if a in used:
            a = (a + random.randint(5, 23)) % max_entity
        b = (a + random.randint(1, 7)) % max_entity
        facts.append((a, b))

    # shuffle facts
    random.shuffle(facts)

    # Serialize: e a r r o w ; tokens
    tokens: List[int] = []
    for a, b in facts:
        tokens.extend([a, vocab.arrow, b, vocab.semi])

    # Scratch steps (optional chain-of-thought hints)
    if random.random() < scratch_fraction:
        tokens.append(vocab.scratch)
        for a,b in chain_edges:
            tokens.extend([a, vocab.arrow, b, vocab.semi])

    # Append query and equal sign sentinel (NO target token in input)
    tokens.extend([vocab.q, start, vocab.arrow, vocab.eq])
    return tokens, target


def batchify(examples: List[List[int]], pad_id: int) -> torch.Tensor:
    L = max(len(x) for x in examples)
    out = torch.full((len(examples), L), pad_id, dtype=torch.long)
    for i, x in enumerate(examples):
        out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return out


# ------------------ Train/eval ------------------
def train_task(model, vocab_size: int, data_fn, device: str, steps=1500, B=32,
               hop_range=(2,5), n_entities=512, pad_id=0, lr=3e-4, warmup=100):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for step in range(steps):
        hops = random.randint(hop_range[0], hop_range[1])
        examples = []
        targets = []
        for _ in range(B):
            start = random.randrange(0, n_entities)
            toks, tgt = data_fn(start, hops)
            examples.append(toks)
            targets.append(tgt)
        x = batchify(examples, pad_id).to(device)
        logits = model(x)
        # Loss on final answer at the <eq> position (last token)
        lengths = torch.tensor([len(t) for t in examples], device=device)
        idx = (lengths - 1)
        # Cross-entropy at final position per example
        gathered = logits[torch.arange(B, device=device), idx]
        y = torch.tensor(targets, device=device)
        loss = F.cross_entropy(gathered, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step < warmup:
            for g in opt.param_groups: g['lr'] = lr * (step + 1) / max(1, warmup)
        


@torch.no_grad()
def eval_task(model, data_fn, device: str, hops_list: List[int], n_entities: int, pad_id: int, trials_per_hop=200) -> Dict[int, float]:
    model.eval()
    acc = {}
    for hops in hops_list:
        correct = 0
        total = 0
        for _ in range(trials_per_hop):
            start = random.randrange(0, n_entities)
            toks, tgt = data_fn(start, hops)
            x = batchify([toks], pad_id).to(device)
            logits = model(x)
            pred = logits[0, x.size(1)-1].argmax().item()
            correct += int(pred == tgt)
            total += 1
        acc[hops] = correct / max(1, total)
    return acc


# ------------------ Main ------------------
def main():
    random.seed(1337); torch.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('results', exist_ok=True)

    # Problem setup
    n_entities = 512
    vocab = make_vocab(n_entities)
    pad_id = vocab.size  # dedicate an extra PAD token outside real symbols

    def data_fn(start, hops):
        toks, tgt = gen_chain_sequence(start, hops, vocab, max_entity=n_entities, distractors=96)
        return toks, tgt

    # Models
    base_cfg = HydraConfig(d=256, n_blocks=8, attn_every=4, moe_experts=4, moe_hidden=256,
                           vocab_size=vocab.size + 1, use_workspace=False, use_pkm=False, fast_ssm=True,
                           vector_moe=True, gate_temp=0.8)
    hydra_off = ToyHydra(base_cfg).to(device)
    hydra_on = ToyHydra(HydraConfig(**{**base_cfg.__dict__, 'use_pkm': True})).to(device)
    trans = BaselineTransformer(d=base_cfg.d, vocab_size=base_cfg.vocab_size, n_layers=base_cfg.n_blocks).to(device)

    print('Params (M): hydra_pkm_off', f'{count_parameters(hydra_off)/1e6:.2f}',
          'hydra_pkm_on', f'{count_parameters(hydra_on)/1e6:.2f}',
          'transformer', f'{count_parameters(trans)/1e6:.2f}')

    # Train all under identical budget
    steps = 1800; B = 48
    print('Training hydra PKM OFF...')
    train_task(hydra_off, vocab.size, lambda s,h: data_fn(s,h), device, steps=steps, B=B,
               hop_range=(2,5), n_entities=n_entities, pad_id=pad_id)
    print('Training hydra PKM ON...')
    train_task(hydra_on, vocab.size, lambda s,h: data_fn(s,h), device, steps=steps, B=B,
               hop_range=(2,5), n_entities=n_entities, pad_id=pad_id)
    print('Training transformer...')
    train_task(trans, vocab.size, lambda s,h: data_fn(s,h), device, steps=steps, B=B,
               hop_range=(2,5), n_entities=n_entities, pad_id=pad_id)

    # Evaluate 2–6 hops
    hops_list = [2,3,4,5,6]
    trials = 300
    results = {
        'hydra_pkm_off': eval_task(hydra_off, lambda s,h: data_fn(s,h), device, hops_list, n_entities, pad_id, trials_per_hop=trials),
        'hydra_pkm_on': eval_task(hydra_on, lambda s,h: data_fn(s,h), device, hops_list, n_entities, pad_id, trials_per_hop=trials),
        'transformer': eval_task(trans, lambda s,h: data_fn(s,h), device, hops_list, n_entities, pad_id, trials_per_hop=trials),
    }

    # Save CSV/JSON
    with open('results/multihop_lookup.json','w') as f:
        json.dump(results, f, indent=2)
    with open('results/multihop_lookup.csv','w', newline='') as f:
        w = csv.writer(f); w.writerow(['model','hops','accuracy'])
        for name, table in results.items():
            for h, acc in table.items():
                w.writerow([name, h, acc])
    print('Saved results: results/multihop_lookup.{json,csv}')


if __name__ == '__main__':
    main()
