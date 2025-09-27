import os
import random
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F

from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters


# ------------------ Config ------------------
@dataclass
class BenchConfig:
    d: int = 224
    n_blocks: int = 8
    vocab_size: int = 2000
    train_steps: int = 1000
    batch_size: int = 64
    lr: float = 3e-4
    warmup: int = 80
    proof_train_len: Tuple[int, int] = (2, 5)  # inclusive
    eval_proofs: List[int] = (2, 3, 4, 5)
    n_eval: int = 400
    n_distractors: int = 6
    seed: int = 4242


# ------------------ Toy tokenizer ------------------
class Vocab:
    def __init__(self, n_atoms: int = 200, extra_tokens: List[str] = None):
        self.tok2id: Dict[str, int] = {'<pad>': 0}
        self.id2tok: List[str] = ['<pad>']
        specials = ['<facts>', '</facts>', '<scratch>', '</scratch>', '<q>', '</q>', '<ans>', '->', ';', '?']
        if extra_tokens:
            specials += extra_tokens
        for s in specials:
            self._add(s)
        for i in range(1, n_atoms + 1):
            self._add(f'A{i}')
    def _add(self, t: str):
        if t not in self.tok2id:
            self.tok2id[t] = len(self.id2tok)
            self.id2tok.append(t)
    def encode(self, toks: List[str]) -> List[int]:
        return [self.tok2id[t] for t in toks]
    def __len__(self):
        return len(self.id2tok)


# ------------------ Data generation ------------------
def make_implication_chain(k: int) -> Tuple[List[Tuple[int, int]], int, int]:
    """Construct A1->A2->...->A{k+1}, query A1->? answer A{k+1}."""
    start = 1
    edges = []
    for i in range(k):
        edges.append((start + i, start + i + 1))
    answer = start + k
    return edges, start, answer


def add_logic_distractors(edges: List[Tuple[int, int]], n_atoms: int, n_distractors: int, avoid: set):
    ds = 0
    while ds < n_distractors:
        u = random.randint(1, n_atoms)
        v = random.randint(1, n_atoms)
        if u == v:
            continue
        if (u, v) in edges or u in avoid or v in avoid:
            continue
        edges.append((u, v))
        ds += 1


def format_example(vocab: Vocab, edges: List[Tuple[int, int]], start_atom: int, answer_atom: int, include_scratch: bool = True) -> Tuple[List[int], int]:
    facts = edges.copy(); random.shuffle(facts)
    toks: List[str] = []
    toks += ['<facts>']
    for (u, v) in facts:
        toks += [f'A{u}', '->', f'A{v}', ';']
    toks += ['</facts>']
    if include_scratch:
        toks += ['<scratch>']
        for (u, v) in edges:
            toks += [f'A{u}', '->', f'A{v}', ';']
        toks += ['</scratch>']
    toks += ['<q>', f'A{start_atom}', '->', '?', '</q>']
    toks += ['<ans>', f'A{answer_atom}']
    x = vocab.encode(toks)
    answer_pos = len(x) - 1
    return x, answer_pos


def batchify(seqs: List[List[int]], answer_pos: List[int], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        y[i, answer_pos[i]] = x[i, answer_pos[i]]
        mask[i, : len(s)] = 1
    return x, y, mask


# ------------------ Train/Eval ------------------
@torch.no_grad()
def evaluate(model, device, vocab: Vocab, k: int, n_samples: int, n_distractors: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_samples):
        edges, start, answer = make_implication_chain(k)
        add_logic_distractors(edges, n_atoms=200, n_distractors=n_distractors, avoid=set([start, answer]))
        x_ids, ap = format_example(vocab, edges, start, answer)
        x = torch.tensor(x_ids, device=device).unsqueeze(0)
        logits = model(x)
        pred = logits[0, ap].argmax(-1).item()
        if pred == vocab.tok2id[f'A{answer}']:
            correct += 1
        total += 1
    return correct / max(1, total)


def train_mixture(model, device, vocab: Vocab, cfg: BenchConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for step in range(cfg.train_steps):
        seqs, ans_pos = [], []
        for _ in range(cfg.batch_size):
            k = random.randint(cfg.proof_train_len[0], cfg.proof_train_len[1])
            edges, start, answer = make_implication_chain(k)
            add_logic_distractors(edges, n_atoms=200, n_distractors=cfg.n_distractors, avoid=set([start, answer]))
            x_ids, ap = format_example(vocab, edges, start, answer)
            seqs.append(x_ids); ans_pos.append(ap)
        x, y, _ = batchify(seqs, ans_pos, pad_id=0)
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        # LR schedule
        if step < cfg.warmup:
            lr_now = cfg.lr * (step + 1) / cfg.warmup
        else:
            progress = (step - cfg.warmup) / max(1, cfg.train_steps - cfg.warmup)
            lr_now = 0.1 * cfg.lr + 0.9 * cfg.lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        for g in opt.param_groups:
            g['lr'] = lr_now
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{cfg.train_steps}: loss={loss.item():.4f}")


# ------------------ Variants ------------------
def build_variants(base_cfg: HydraConfig):
    variants = {}
    variants['transformer'] = BaselineTransformer(d=base_cfg.d, vocab_size=base_cfg.vocab_size, n_layers=base_cfg.n_blocks)
    cfg_off = HydraConfig(**{**base_cfg.__dict__, 'use_workspace': False})
    variants['hydra_ws_off'] = ToyHydra(cfg_off)
    cfg_on = HydraConfig(**{**base_cfg.__dict__, 'use_workspace': True})
    variants['hydra_ws_on'] = ToyHydra(cfg_on)
    return variants


# ------------------ Main ------------------
if __name__ == '__main__':
    cfg = BenchConfig()
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = Vocab(n_atoms=200)
    base_cfg = HydraConfig(d=cfg.d, n_blocks=cfg.n_blocks, vocab_size=max(cfg.vocab_size, len(vocab)), moe_experts=4, moe_hidden=224, chunk_size=32, fast_ssm=True, disable_moe=False, disable_attn=False, vector_moe=True)

    if os.environ.get('BENCH_QUICK'):
        cfg.train_steps = 6
        cfg.batch_size = 8
        cfg.n_eval = 24
        cfg.eval_proofs = (2, 3)
        print('[quick] using small settings:', cfg)

    variants = build_variants(base_cfg)
    print('Params (M):')
    for name, model in variants.items():
        print(f"  {name:16s}: {count_parameters(model)/1e6:.2f}")

    os.makedirs('results', exist_ok=True)

    for name, model in variants.items():
        print(f"\nTraining {name} ...")
        model.to(device)
        train_mixture(model, device, vocab, cfg)
        model.to('cpu'); torch.cuda.empty_cache() if device.startswith('cuda') else None

    rows = []
    for name, model in variants.items():
        model.to(device).eval()
        for k in cfg.eval_proofs:
            acc = evaluate(model, device, vocab, k=k, n_samples=cfg.n_eval, n_distractors=cfg.n_distractors)
            print(f"{name} proof_len={k}: acc={acc:.3f}")
            rows.append({'model': name, 'proof_len': k, 'accuracy': acc, 'n': cfg.n_eval})
        model.to('cpu'); torch.cuda.empty_cache() if device.startswith('cuda') else None

    with open('results/logic_accuracy.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'proof_len', 'accuracy', 'n'])
        writer.writeheader(); writer.writerows(rows)
    print('Saved results/logic_accuracy.csv')

    try:
        import matplotlib.pyplot as plt
        from collections import defaultdict
        grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for r in rows:
            grouped[r['model']].append((r['proof_len'], r['accuracy']))
        plt.figure()
        for m, pts in grouped.items():
            pts = sorted(pts)
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=m)
        plt.xlabel('Proof length (steps)')
        plt.ylabel('Accuracy')
        plt.title('Logic Composition: Accuracy vs Proof Length')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/fig_logic_accuracy.png', dpi=150)
        print('Saved results/fig_logic_accuracy.png')
    except Exception as e:
        print('Plotting failed:', e)
