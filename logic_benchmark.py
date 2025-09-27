import math, time, csv, json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters
import random
import copy

# Logic benchmark vocab
vocab = ['<pad>', '<eos>'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['->', '.', '?']
vocab_size = len(vocab)
token_to_id = {t: i for i, t in enumerate(vocab)}
id_to_token = {i: t for t, i in token_to_id.items()}

def tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        if s[i:i+2] == '->':
            tokens.append('->')
            i += 2
        elif s[i] in token_to_id:
            tokens.append(s[i])
            i += 1
        else:
            i += 1  # skip invalid
    return [token_to_id[t] for t in tokens if t in token_to_id]

def generate_example(k, num_distractors=5):
    vars_list = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    chain_vars = vars_list[:k+1]
    distractor_vars = vars_list[k+1:]
    premises = [f'{chain_vars[i]}->{chain_vars[i+1]}' for i in range(k)]
    distractors = []
    for _ in range(num_distractors):
        a = random.choice(distractor_vars)
        b = random.choice(distractor_vars)
        if a != b:
            distractors.append(f'{a}->{b}')
    all_premises = premises + distractors
    random.shuffle(all_premises)
    premises_str = '.'.join(all_premises) + '.'
    query = f'{chain_vars[0]}->'
    full = premises_str + query
    tokens = tokenize(full)
    target = chain_vars[k]
    target_id = token_to_id[target]
    return tokens, target_id

# Training function for logic dataset
def train_logic(model, train_data, device, steps=2000, B=8, lr=3e-4, warmup=40):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        batch = random.sample(train_data, min(B, len(train_data)))
        max_len = max(len(seq) for seq in batch)
        x_batch = []
        y_batch = []
        for seq in batch:
            x = seq[:-1]
            y = seq[1:]
            pad_len = max_len - len(x)
            x = x + [token_to_id['<pad>']] * pad_len
            y = y + [token_to_id['<pad>']] * pad_len
            x_batch.append(x)
            y_batch.append(y)
        x = torch.tensor(x_batch, device=device)
        y = torch.tensor(y_batch, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=token_to_id['<pad>'])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step < warmup:
            for g in opt.param_groups:
                g['lr'] = lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / (steps - warmup)
            for g in opt.param_groups:
                g['lr'] = 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))
        if (step + 1) % 50 == 0:
            losses.append((step + 1, float(loss.detach())))
    return losses

# Evaluation function
@torch.no_grad()
def evaluate_logic(model, test_data, device):
    model.eval()
    accuracies = {}
    for k, examples in test_data.items():
        correct = 0
        total = 0
        for tokens, target in examples:
            x = torch.tensor(tokens[:-1], device=device).unsqueeze(0)
            logits = model(x)
            pred = logits[0, -1].argmax().item()
            if pred == target:
                correct += 1
            total += 1
        accuracies[k] = correct / total if total > 0 else 0.0
    return accuracies

# Main
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_cfg = HydraConfig(d=256, n_blocks=8, attn_every=4, moe_experts=4, moe_hidden=256, vocab_size=vocab_size, use_workspace=False, use_pkm=False)
    
    cfg_on = copy.deepcopy(base_cfg)
    cfg_on.use_workspace = True
    variants = {
        'transformer': BaselineTransformer(d=256, vocab_size=vocab_size, n_layers=8),
        'hydra_workspace_off': ToyHydra(base_cfg),
        'hydra_workspace_on': ToyHydra(cfg_on)
    }
    
    # Generate train data
    train_data = []
    for k in range(2, 6):
        for _ in range(2000):
            tokens, _ = generate_example(k)
            train_data.append(tokens)
    random.shuffle(train_data)
    
    # Generate test data
    test_data = {k: [] for k in range(2, 6)}
    for k in range(2, 6):
        for _ in range(100):
            tokens, target = generate_example(k)
            test_data[k].append((tokens, target))
    
    # Train and evaluate
    results = {}
    for name, model in variants.items():
        print(f'Training {name}...')
        losses = train_logic(model.to(device), train_data, device, steps=2000, B=16)
        accuracies = evaluate_logic(model, test_data, device)
        results[name] = {'losses': losses, 'accuracies': accuracies}
        model.to('cpu')
        torch.cuda.empty_cache() if device.startswith('cuda') else None
    
    # Save losses
    with open('results/logic_train_losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'step', 'loss'])
        for name, data in results.items():
            for step, loss in data['losses']:
                writer.writerow([name, step, loss])
    
    # Save accuracies
    with open('results/logic_accuracies.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'proof_length', 'accuracy'])
        for name, data in results.items():
            for k, acc in data['accuracies'].items():
                writer.writerow([name, k, acc])
    
    print('Saved logic_train_losses.csv and logic_accuracies.csv')
