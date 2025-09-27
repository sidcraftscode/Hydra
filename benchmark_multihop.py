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
    train_steps: int = 1800
    batch_size: int = 64
    lr: float = 2e-4
    warmup: int = 200
    max_hops_train: Tuple[int, int] = (2, 6)  # inclusive
    eval_hops: List[int] = (2, 3, 4, 5, 6)
    n_eval: int = 400
    n_distractors: int = 12
    seed: int = 1337


# ------------------ Toy tokenizer ------------------
class Vocab:
    def __init__(self, n_entities: int = 200, extra_tokens: List[str] = None):
        # Reserve 0 for padding
        self.tok2id: Dict[str, int] = {'<pad>': 0}
        self.id2tok: List[str] = ['<pad>']
        specials = ['<facts>', '</facts>', '<scratch>', '</scratch>', '<q>', '</q>', '<ans>', '->', ';', '?']
        if extra_tokens:
            specials += extra_tokens
        for s in specials:
            self._add(s)
        for i in range(1, n_entities + 1):
            self._add(f'E{i}')
    def _add(self, t: str):
        if t not in self.tok2id:
            self.tok2id[t] = len(self.id2tok)
            self.id2tok.append(t)
    def encode(self, toks: List[str]) -> List[int]:
        return [self.tok2id[t] for t in toks]
    def __len__(self):
        return len(self.id2tok)


# ------------------ Data generation ------------------
def make_chain(k: int, n_entities: int = 200) -> Tuple[List[Tuple[int, int]], int, int]:
    """Return list of directed edges (u->v) forming chain length k, start entity id, answer entity id.
    Start is randomized to avoid mapping hops->answer trivially."""
    start = random.randint(1, max(1, n_entities - k))
    edges = []
    for i in range(k):
        edges.append((start + i, start + i + 1))
    answer = start + k
    return edges, start, answer


def add_distractors(edges: List[Tuple[int, int]], n_ents: int, n_distractors: int, avoid: set, chain_nodes: set = None):
    """Add random distractors; half branch out from chain nodes to increase ambiguity."""
    ds = 0
    half = n_distractors // 2
    chain_nodes = chain_nodes or set()
    # Branching distractors from chain nodes
    while ds < half:
        if not chain_nodes:
            break
        u = random.choice(list(chain_nodes))
        v = random.randint(1, n_ents)
        if u == v: continue
        if (u, v) in edges: continue
        edges.append((u, v)); ds += 1
    # Pure random distractors
    while ds < n_distractors:
        u = random.randint(1, n_ents)
        v = random.randint(1, n_ents)
        if u == v: continue
        if (u, v) in edges: continue
        if u in avoid and v in avoid: continue
        edges.append((u, v)); ds += 1


def format_example(vocab: Vocab, edges: List[Tuple[int, int]], start_ent: int, answer_ent: int, include_scratch: bool = True) -> Tuple[List[int], int, int]:
    """Return tokenized example, position to predict (the <ans> index), and target token id.
    Layout: <facts> ... </facts> <scratch> ... </scratch> <q> E# -> ? </q> <ans>
    The model must predict E{answer} as the NEXT token after <ans> (standard causal LM shift)."""
    # Shuffle facts (multi-hop lookup requires unordered KB)
    facts = edges.copy()
    random.shuffle(facts)
    toks: List[str] = []
    toks += ['<facts>']
    for (u, v) in facts:
        toks += [f'E{u}', '->', f'E{v}', ';']
    toks += ['</facts>']
    if include_scratch:
        toks += ['<scratch>']
        # Provide step scaffolding (not used for supervision, but helps learn)
        for (u, v) in edges:  # in original chain order
            toks += [f'E{u}', '->', f'E{v}', ';']
        toks += ['</scratch>']
    toks += ['<q>', f'E{start_ent}', '->', '?', '</q>']
    toks += ['<ans>']
    x = vocab.encode(toks)
    ans_pos = len(x) - 1  # index of <ans>
    target_id = vocab.tok2id[f'E{answer_ent}']
    return x, ans_pos, target_id


def batchify(seqs: List[List[int]], answer_pos: List[int], targets: List[int], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        # target only at answer position (predict next token after <ans>)
        y[i, answer_pos[i]] = targets[i]
        mask[i, : len(s)] = 1
    return x, y, mask


# ------------------ Train/Eval ------------------
@torch.no_grad()
def evaluate(model, device, vocab: Vocab, hops: int, n_samples: int, n_distractors: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_samples):
        edges, start, answer = make_chain(hops, n_entities=200)
        chain_nodes = {u for (u, _v) in edges} | {answer}
        # For short hops, avoid branching from start and reduce distractors
        if hops <= 2:
            chain_nodes = {n for n in chain_nodes if n != start}
            local_n = max(2, n_distractors // 2)
        else:
            local_n = n_distractors
        add_distractors(edges, n_ents=200, n_distractors=local_n, avoid=set(), chain_nodes=chain_nodes)
        x_ids, ans_pos, target_id = format_example(vocab, edges, start, answer)
        x = torch.tensor(x_ids, device=device).unsqueeze(0)
        logits = model(x)
        pred = logits[0, ans_pos].argmax(-1).item()
        if pred == target_id:
            correct += 1
        total += 1
    return correct / max(1, total)


def train_mixture(model, device, vocab: Vocab, cfg: BenchConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for step in range(cfg.train_steps):
        # Sample mixed hops in batch
        seqs, ans_pos, targets = [], [], []
        for _ in range(cfg.batch_size):
            k = random.randint(cfg.max_hops_train[0], cfg.max_hops_train[1])
            edges, start, answer = make_chain(k, n_entities=200)
            chain_nodes = {u for (u, _v) in edges} | {answer}
            if k <= 2:
                chain_nodes = {n for n in chain_nodes if n != start}
                local_n = max(2, cfg.n_distractors // 2)
            else:
                local_n = cfg.n_distractors
            add_distractors(edges, n_ents=200, n_distractors=local_n, avoid=set(), chain_nodes=chain_nodes)
            x_ids, ap, tgt = format_example(vocab, edges, start, answer)
            seqs.append(x_ids)
            ans_pos.append(ap)
            targets.append(tgt)
        x, y, _ = batchify(seqs, ans_pos, targets, pad_id=0)
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        # MoE aux loss (if available)
    if hasattr(model, 'moe_stats') and model.moe_stats:
            lb_terms = []
            for stat in model.moe_stats:
                usage = stat.get('usage', None)
                if usage is not None:
                    E = usage.numel()
                    lb = (E * (usage ** 2).sum() - 1.0)
                    lb_terms.append(lb)
            if lb_terms:
        w = getattr(getattr(model, 'cfg', object()), 'aux_load_balance_weight', 0.02)
        loss = loss + w * torch.stack(lb_terms).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Simple LR warmup + cosine
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
    # Baseline Transformer
    variants['transformer'] = BaselineTransformer(d=base_cfg.d, vocab_size=base_cfg.vocab_size, n_layers=base_cfg.n_blocks)
    # Hydra PKM OFF
    cfg_off = HydraConfig(**{**base_cfg.__dict__, 'use_pkm': False})
    variants['hydra_pkm_off'] = ToyHydra(cfg_off)
    # Hydra PKM ON
    cfg_on = HydraConfig(**{**base_cfg.__dict__, 'use_pkm': True})
    variants['hydra_pkm_on'] = ToyHydra(cfg_on)
    return variants


# ------------------ Main ------------------
if __name__ == '__main__':
    cfg = BenchConfig()
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Vocab sized to config
    vocab = Vocab(n_entities=200)
    base_cfg = HydraConfig(
        d=cfg.d,
        n_blocks=cfg.n_blocks,
        vocab_size=max(cfg.vocab_size, len(vocab)),
        moe_experts=6,
        moe_hidden=288,
        chunk_size=32,
        fast_ssm=True,
        disable_moe=False,
        disable_attn=False,
        vector_moe=True,
        n_heads=4,
        attn_every=2,
        gate_temp=0.8,
        ssm_kernel=12,
        aux_load_balance_weight=0.02,
    )

    # Quick mode for CI/smoke test
    if os.environ.get('BENCH_QUICK'):
        cfg.train_steps = 6
        cfg.batch_size = 8
        cfg.n_eval = 24
        cfg.eval_hops = (2, 3)
        print('[quick] using small settings:', cfg)

    variants = build_variants(base_cfg)
    print('Params (M):')
    for name, model in variants.items():
        print(f"  {name:16s}: {count_parameters(model)/1e6:.2f}")

    os.makedirs('results', exist_ok=True)

    # Train all variants on same task mixture
    for name, model in variants.items():
        print(f"\nTraining {name} ...")
        model.to(device)
        train_mixture(model, device, vocab, cfg)
        model.to('cpu'); torch.cuda.empty_cache() if device.startswith('cuda') else None

    # Evaluate accuracy vs hops
    rows = []
    for name, model in variants.items():
        model.to(device).eval()
        for h in cfg.eval_hops:
            acc = evaluate(model, device, vocab, hops=h, n_samples=cfg.n_eval, n_distractors=cfg.n_distractors)
            print(f"{name} hops={h}: acc={acc:.3f}")
            rows.append({'model': name, 'hops': h, 'accuracy': acc, 'n': cfg.n_eval})
        model.to('cpu'); torch.cuda.empty_cache() if device.startswith('cuda') else None

    # Save CSV
    with open('results/multihop_accuracy.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'hops', 'accuracy', 'n'])
        writer.writeheader(); writer.writerows(rows)
    print('Saved results/multihop_accuracy.csv')

    # Plot
    try:
        import matplotlib.pyplot as plt
        from collections import defaultdict
        grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for r in rows:
            grouped[r['model']].append((r['hops'], r['accuracy']))
        plt.figure()
        for m, pts in grouped.items():
            pts = sorted(pts)
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=m)
        plt.xlabel('Hops')
        plt.ylabel('Accuracy')
        plt.title('Multi-hop Lookup: Accuracy vs Hops')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/fig_multihop_accuracy.png', dpi=150)
        print('Saved results/fig_multihop_accuracy.png')
    except Exception as e:
        print('Plotting failed:', e)
