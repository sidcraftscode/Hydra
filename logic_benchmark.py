import csv
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters


# ------------------ Data: Implication chains ------------------

@dataclass
class LogicVocab:
    # Symbols: entities A..Z, arrow, comma/sep, EOS, PAD
    entities: List[str]
    arrow: str = '->'
    sep: str = ','
    query_tok: str = '?'
    eos: str = '<eos>'
    pad: str = '<pad>'

    def build(self) -> Tuple[Dict[str, int], List[str]]:
        toks = [self.pad, self.eos, self.arrow, self.sep, self.query_tok] + self.entities
        itos = toks
        stoi = {t: i for i, t in enumerate(itos)}
        return stoi, itos


def make_entities(n: int) -> List[str]:
    base = [chr(ord('A') + i) for i in range(26)]
    out = []
    k = 0
    while len(out) < n:
        t = base[k % 26] if k < 26 else base[k % 26] + str(k // 26)
        out.append(t)
        k += 1
    return out


def gen_chain_example(stoi: Dict[str, int], L: int, max_branch: int = 0) -> Tuple[List[int], int]:
    # Chain: A->B, B->C, ..., X->Y ; query: A->? ; target=Y
    # Optionally add unused branches to avoid trivial memorization (max_branch distractors)
    ents = [e for e in stoi.keys() if e not in ['<pad>', '<eos>', '->', ',', '?']]
    ents = [e for e in ents if not e.startswith('<')]
    # sample unique chain of length L+1 nodes
    nodes = random.sample(ents, L + 1)
    pairs = [(nodes[i], nodes[i + 1]) for i in range(L)]
    distractors = []
    for _ in range(max_branch):
        x, y = random.sample(ents, 2)
        if (x, y) not in pairs:
            distractors.append((x, y))
    seq_tokens: List[int] = []
    for (a, b) in pairs + distractors:
        seq_tokens.extend([stoi[a], stoi['->'], stoi[b], stoi[',']])
    # query
    seq_tokens.extend([stoi[nodes[0]], stoi['->'], stoi['?'], stoi['<eos>']])
    target = stoi[nodes[-1]]
    return seq_tokens, target


def batchify(examples: List[Tuple[List[int], int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _ in examples)
    B = len(examples)
    X = torch.full((B, max_len), pad_id, dtype=torch.long)
    Y = torch.empty(B, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        X[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        Y[i] = y
    return X, Y


# ------------------ Training/eval ------------------

@torch.no_grad()
def evaluate(model, stoi: Dict[str, int], device: str, lengths=(2, 3, 4, 5), n_per_len=200) -> Dict[int, float]:
    model.eval()
    pad = stoi['<pad>']
    accs = {}
    # reset workspace state between batches for fairness
    def reset_ws(m):
        if hasattr(m, 'workspace') and m.workspace is not None and hasattr(m.workspace, 'reset_state'):
            m.workspace.reset_state()
    for L in lengths:
        correct = 0
        total = 0
        for _ in range(n_per_len // 20):
            batch = [gen_chain_example(stoi, L, max_branch=1) for _ in range(20)]
            X, Y = batchify(batch, pad)
            X = X.to(device)
            Y = Y.to(device)
            reset_ws(model)
            logits = model(X)
            # take last non-pad position logit as answer distribution; here it's fixed at <eos> - 1 position
            # The answer should be generated at the token after '?', i.e., logits at position of <eos> - 1 gives next token
            lengths_tok = (X != pad).sum(dim=1)
            idxs = lengths_tok - 1  # position of <eos>
            next_logits = logits[torch.arange(X.size(0), device=device), idxs - 0]  # logits at <eos> to predict <eos> token; but we want token before <eos>
            # Instead, predict token at position of '?' (which is at eos-1); so use eos-1 logits to predict '?', not helpful.
            # Better: shift targets: train to predict next token always. We'll hack: get logits at position where '?' is, and read next-token distribution.
            # Find '?' positions per row
            qm = (X == stoi['?'])
            qpos = qm.int().argmax(dim=1)  # first/only '?'
            next_logits = logits[torch.arange(X.size(0), device=device), qpos]
            pred = next_logits.argmax(dim=-1)
            correct += (pred == Y).sum().item()
            total += X.size(0)
        accs[L] = correct / max(1, total)
    return accs


def train_on_synth(model, stoi: Dict[str, int], device: str, steps=2000, B=32, L_train=(2, 3, 4), lr=3e-4):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pad = stoi['<pad>']
    for step in range(steps):
        batch = [gen_chain_example(stoi, random.choice(L_train), max_branch=1) for _ in range(B)]
        X, Y_ans = batchify(batch, pad)
        X = X.to(device)
        Y_ans = Y_ans.to(device)
        # Train with next-token prediction objective at '?' position to output correct entity
        # Reset workspace state per batch for fairness (scratchpad, not cross-example memory)
        if hasattr(model, 'workspace') and model.workspace is not None and hasattr(model.workspace, 'reset_state'):
            model.workspace.reset_state()
        logits = model(X)
        qm = (X == stoi['?'])
        qpos = qm.int().argmax(dim=1)
        logits_q = logits[torch.arange(B, device=device), qpos]
        loss = F.cross_entropy(logits_q, Y_ans)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            print(f"step {step+1}: loss={loss.item():.4f}")


# ------------------ Build models ------------------

def build_models(vocab_size: int, base_d=192, blocks=6) -> Dict[str, torch.nn.Module]:
    # Slightly reduce d/blocks so training is quick while preserving Hydra advantages
    base_cfg = HydraConfig(d=base_d, n_blocks=blocks, attn_every=3, moe_experts=4, moe_hidden=192, vocab_size=vocab_size,
                           fast_ssm=True, vector_moe=True, gate_temp=0.8)
    # Workspace ON
    hydra_ws_on = ToyHydra(HydraConfig(**{**base_cfg.__dict__, 'use_workspace': True}))
    # Workspace OFF
    hydra_ws_off = ToyHydra(HydraConfig(**{**base_cfg.__dict__, 'use_workspace': False}))
    # Transformer baseline with similar params
    transformer = BaselineTransformer(d=base_d, vocab_size=vocab_size, n_layers=blocks)
    return {
        'hydra_workspace_on': hydra_ws_on,
        'hydra_workspace_off': hydra_ws_off,
        'transformer': transformer,
    }


# ------------------ Main: run benchmark ------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Vocab with 200 entities to avoid collisions
    entities = make_entities(200)
    vocab = LogicVocab(entities=entities)
    stoi, itos = vocab.build()

    models = build_models(vocab_size=len(itos))
    for k, m in models.items():
        m.to(device)
        print(f"{k:22s} params: {count_parameters(m)/1e6:.2f}M")

    # Train each model briefly on lengths 2-4; hold 5 for evaluation stretch
    train_steps = int(os.getenv('LOGIC_TRAIN_STEPS', '1500'))
    for name in ['hydra_workspace_on', 'hydra_workspace_off', 'transformer']:
        print(f"\nTraining {name}...")
        train_on_synth(models[name], stoi, device, steps=train_steps, B=32, L_train=(2, 3, 4), lr=3e-4)

    # Evaluate accuracy vs proof length
    lengths = (2, 3, 4, 5)
    rows = []
    for name, model in models.items():
        n_per_len = int(os.getenv('LOGIC_EVAL_PER_LEN', '200'))
        accs = evaluate(model, stoi, device, lengths=lengths, n_per_len=n_per_len)
        print(f"{name} accuracy:", {L: f"{accs[L]:.3f}" for L in lengths})
        for L in lengths:
            rows.append([name, L, accs[L]])

    # Save CSV
    os.makedirs('results', exist_ok=True)
    with open('results/logic_composition_accuracy.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'proof_length', 'accuracy'])
        w.writerows(rows)
    print("Saved results/logic_composition_accuracy.csv")


if __name__ == '__main__':
    main()
