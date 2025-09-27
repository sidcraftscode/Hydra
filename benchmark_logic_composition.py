"""Logic composition benchmark (implication chains 2–5 steps).

Data: Chains like A->B, B->C, ... with distractors; query A->?
Run: Hydra workspace ON vs OFF vs Transformer.
Output: Accuracy vs proof length (CSV/JSON in results/).
"""
from __future__ import annotations
import os, random, json, csv
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F

from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters


# ------------------ Vocab ------------------
@dataclass
class LogicVocab:
    n_atoms: int
    implies: int
    semi: int
    query: int
    scratch: int
    eq: int
    @property
    def size(self):
        return self.eq + 1

def make_vocab(n_atoms: int = 512) -> LogicVocab:
    return LogicVocab(n_atoms, n_atoms, n_atoms+1, n_atoms+2, n_atoms+3, n_atoms+4)


# ------------------ Data ------------------
def gen_implication_chain(start: int, length: int, vocab: LogicVocab, max_atom: int,
                          distractors: int = 128, interleave_noise: bool = True, scratch_fraction: float = 0.6) -> Tuple[List[int], int]:
    chain: List[Tuple[int,int]] = []
    cur = start
    used = {cur}
    for _ in range(length):
        for _try in range(8):
            nxt = random.randrange(0, max_atom)
            if nxt not in used:
                break
        chain.append((cur, nxt))
        cur = nxt
        used.add(cur)
    target = cur

    facts: List[Tuple[int,int]] = chain.copy()
    # draw random unrelated implications as distractors
    seen = set([a for a,_ in chain] + [b for _,b in chain])
    for _ in range(distractors):
        a = random.randrange(0, max_atom)
        if a in seen:
            a = (a + random.randint(9, 29)) % max_atom
        b = (a + random.randint(1, 9)) % max_atom
        facts.append((a, b))

    # serialize rules, possibly interleaving noise with parts of the true chain
    tokens: List[int] = []
    if interleave_noise:
        mixed = []
        # interleave: chunks of true chain with noise facts
        true_idx = 0
        noise = [f for f in facts if f not in chain]
        random.shuffle(noise)
        for _ in range(len(chain)):
            mixed.append(chain[true_idx]); true_idx += 1
            # add some noise in between
            for _ in range(3):
                if noise:
                    mixed.append(noise.pop())
        mixed.extend(noise)
    else:
        mixed = facts
    random.shuffle(mixed)

    for a, b in mixed:
        tokens.extend([a, vocab.implies, b, vocab.semi])

    # optional chain-of-thought scratch revealing the proof steps which the workspace can cache
    if random.random() < scratch_fraction:
        tokens.append(vocab.scratch)
        cur = start
        for _ in range(length):
            # replay the true chain from 'chain'
            nxt = chain[_][1]
            tokens.extend([cur, vocab.implies, nxt, vocab.semi])
            cur = nxt

    # query without the answer token; answer is predicted at the <eq> position
    tokens.extend([vocab.query, start, vocab.implies, vocab.eq])
    return tokens, target


def batchify(examples: List[List[int]], pad_id: int) -> torch.Tensor:
    L = max(len(x) for x in examples)
    out = torch.full((len(examples), L), pad_id, dtype=torch.long)
    for i, x in enumerate(examples):
        out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return out


# ------------------ Train/eval ------------------
def train_task(model, vocab_size: int, data_fn, device: str, steps=1600, B=48, chain_range=(2,5), n_atoms=512, pad_id=0, lr=3e-4, warmup=120):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for step in range(steps):
        length = random.randint(chain_range[0], chain_range[1])
        examples = []
        targets = []
        for _ in range(B):
            start = random.randrange(0, n_atoms)
            toks, tgt = data_fn(start, length)
            examples.append(toks)
            targets.append(tgt)
        x = batchify(examples, pad_id).to(device)
        logits = model(x)
        lengths = torch.tensor([len(t) for t in examples], device=device)
        idx = (lengths - 1)
        gathered = logits[torch.arange(B, device=device), idx]
        y = torch.tensor(targets, device=device)
        loss = F.cross_entropy(gathered, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step < warmup:
            for g in opt.param_groups: g['lr'] = lr * (step + 1) / max(1, warmup)


@torch.no_grad()
def eval_task(model, data_fn, device: str, lengths: List[int], n_atoms: int, pad_id: int, trials_per_len=300) -> Dict[int, float]:
    model.eval()
    acc = {}
    for L in lengths:
        correct = 0
        total = 0
        for _ in range(trials_per_len):
            start = random.randrange(0, n_atoms)
            toks, tgt = data_fn(start, L)
            x = batchify([toks], pad_id).to(device)
            logits = model(x)
            pred = logits[0, x.size(1)-1].argmax().item()
            correct += int(pred == tgt)
            total += 1
        acc[L] = correct / max(1, total)
    return acc


def main():
    random.seed(2025); torch.manual_seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('results', exist_ok=True)

    n_atoms = 512
    vocab = make_vocab(n_atoms)
    pad_id = vocab.size  # dedicated PAD token

    def data_fn(start, length):
        return gen_implication_chain(start, length, vocab, max_atom=n_atoms, distractors=96, interleave_noise=True)

    # Models (same capacity/budget except workspace switch)
    base_cfg = HydraConfig(d=256, n_blocks=8, attn_every=4, moe_experts=4, moe_hidden=256,
                           vocab_size=vocab.size + 1, use_workspace=False, use_pkm=False, fast_ssm=True,
                           vector_moe=True, gate_temp=0.8)
    hydra_off = ToyHydra(base_cfg).to(device)
    hydra_on = ToyHydra(HydraConfig(**{**base_cfg.__dict__, 'use_workspace': True})).to(device)
    trans = BaselineTransformer(d=base_cfg.d, vocab_size=base_cfg.vocab_size, n_layers=base_cfg.n_blocks).to(device)

    print('Params (M): hydra_ws_off', f'{count_parameters(hydra_off)/1e6:.2f}',
          'hydra_ws_on', f'{count_parameters(hydra_on)/1e6:.2f}',
          'transformer', f'{count_parameters(trans)/1e6:.2f}')

    steps = 1800; B = 48
    print('Training hydra WS OFF...')
    train_task(hydra_off, vocab.size, data_fn, device, steps=steps, B=B, chain_range=(2,5), n_atoms=n_atoms, pad_id=pad_id)
    print('Training hydra WS ON...')
    train_task(hydra_on, vocab.size, data_fn, device, steps=steps, B=B, chain_range=(2,5), n_atoms=n_atoms, pad_id=pad_id)
    print('Training transformer...')
    train_task(trans, vocab.size, data_fn, device, steps=steps, B=B, chain_range=(2,5), n_atoms=n_atoms, pad_id=pad_id)

    lengths = [2,3,4,5]
    trials = 300
    results = {
        'hydra_ws_off': eval_task(hydra_off, data_fn, device, lengths, n_atoms, pad_id, trials_per_len=trials),
        'hydra_ws_on': eval_task(hydra_on, data_fn, device, lengths, n_atoms, pad_id, trials_per_len=trials),
        'transformer': eval_task(trans, data_fn, device, lengths, n_atoms, pad_id, trials_per_len=trials),
    }

    with open('results/logic_composition.json','w') as f:
        json.dump(results, f, indent=2)
    with open('results/logic_composition.csv','w', newline='') as f:
        w = csv.writer(f); w.writerow(['model','proof_len','accuracy'])
        for name, table in results.items():
            for L, acc in table.items():
                w.writerow([name, L, acc])
    print('Saved results: results/logic_composition.{json,csv}')


if __name__ == '__main__':
    main()
