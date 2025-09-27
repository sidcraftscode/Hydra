import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import os
import random
import numpy as np
from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters

# ------------------ Data Generation ------------------

def generate_arithmetic_data(num_samples=1000, max_val=100):
    """Generates simple arithmetic problems like 'a + b = c'."""
    operations = ['+', '-', '*']
    data = []
    for _ in range(num_samples):
        op = random.choice(operations)
        a, b = random.randint(0, max_val), random.randint(0, max_val)
        if op == '-':
            # Ensure result is not negative
            a, b = max(a, b), min(a, b)
        
        question = f"{a} {op} {b}"
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else: # '*'
            answer = a * b
            
        data.append({'question': question, 'answer': answer, 'op': op})
    return data

def tokenize_data(data, vocab):
    """Converts string data to token IDs."""
    tokenized_data = []
    for item in data:
        question_tokens = [vocab.get(c, 0) for c in item['question']]
        # For this toy benchmark, we'll predict the answer directly.
        # A more complex setup would predict token by token.
        answer_token = item['answer'] # We'll treat the answer as a single "token" for simplicity
        tokenized_data.append({
            'question': torch.tensor(question_tokens, dtype=torch.long),
            'answer': torch.tensor(answer_token, dtype=torch.long),
            'op': item['op']
        })
    return tokenized_data

def build_vocab():
    """Builds a simple character-level vocabulary."""
    chars = "0123456789+-* ="
    vocab = {c: i+1 for i, c in enumerate(chars)}
    vocab['<pad>'] = 0
    # For simplicity, answers will be mapped to a separate output space
    return vocab

# ------------------ Models ------------------

class DenseEquivalent(nn.Module):
    """A dense model with parameter count similar to the MoE model."""
    def __init__(self, d_model, vocab_size, n_layers, ffn_mult):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=4, ffn_mult=ffn_mult) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # The head will predict the numerical answer directly
        self.head = nn.Linear(d_model, 100*100 + 1) # Max answer for 100*100

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        # Use the representation of the last token to predict the answer
        x = self.ln_f(x[:, -1, :])
        return self.head(x)

def get_models(vocab_size, device):
    """Get MoE and dense models with similar parameter counts."""
    moe_cfg = HydraConfig(
        d=128,
        vocab_size=vocab_size,
        n_blocks=4,
        attn_every=2,
        moe_on=list(range(0, 4, 2)),
        moe_experts=8,
        moe_hidden=128,
        fast_ssm=True,
        disable_attn=True # Focus on MoE vs Dense FFN
    )
    moe_hydra = ToyHydra(moe_cfg)
    moe_hydra.predict_last_only = True
    moe_hydra.head = nn.Linear(moe_cfg.d, 100*100 + 1)
    
    # Adjust dense model to have similar params
    moe_params = count_parameters(moe_hydra)
    
    # Create a dense model and adjust its FFN multiplier to match params
    dense_model = DenseEquivalent(d_model=128, vocab_size=vocab_size, n_layers=4, ffn_mult=4)
    dense_params = count_parameters(dense_model)

    # Iteratively find a good multiplier
    ffn_mult = 4
    for _ in range(10):
        if abs(dense_params - moe_params) < 1e5:
            break
        ffn_mult = int(ffn_mult * (moe_params / dense_params))
        dense_model = DenseEquivalent(d_model=128, vocab_size=vocab_size, n_layers=4, ffn_mult=ffn_mult)
        dense_params = count_parameters(dense_model)

    models = {
        "hydra_moe": moe_hydra.to(device),
        "dense_equivalent": dense_model.to(device)
    }
    print(f"MoE Model Params: {moe_params/1e6:.2f}M")
    print(f"Dense Model Params: {dense_params/1e6:.2f}M (FFN mult: {ffn_mult})")
    
    return models

# ------------------ Training & Evaluation ------------------

def train(model, data, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for item in data:
        question = item['question'].unsqueeze(0).to(device)
        answer = item['answer'].unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        logits = model(question)
        loss = criterion(logits, answer)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)

@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    correct = 0
    total_latency = 0
    
    for item in data:
        question = item['question'].unsqueeze(0).to(device)
        answer = item['answer']
        
        # Warmup not included in timing
        _ = model(question)
        
        if device.startswith('cuda'): torch.cuda.synchronize()
        t0 = time.time()
        logits = model(question)
        if device.startswith('cuda'): torch.cuda.synchronize()
        total_latency += (time.time() - t0)
        
        pred = logits.argmax(dim=-1).item()
        
        if pred == answer:
            correct += 1
            
    accuracy = correct / len(data)
    avg_latency_ms = (total_latency / len(data)) * 1000
    tokens_per_sec = len(data) / total_latency if total_latency > 0 else 0
    
    return accuracy, avg_latency_ms, tokens_per_sec

# ------------------ Main Benchmark Loop ------------------

def conditional_compute_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # --- Config ---
    epochs = 5
    lr = 0.001
    num_samples = 2000
    
    # --- Data ---
    vocab = build_vocab()
    data = generate_arithmetic_data(num_samples=num_samples)
    tokenized_data = tokenize_data(data, vocab)
    
    # --- Models ---
    models = get_models(len(vocab) + 1, device)
    
    results = []

    for name, model in models.items():
        print(f"--- Benchmarking {name} ---")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # --- Training ---
        for epoch in range(epochs):
            random.shuffle(tokenized_data)
            loss = train(model, tokenized_data, optimizer, criterion, device)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
            
        # --- Evaluation ---
        accuracy, latency, throughput = evaluate(model, tokenized_data, device)
        
        results.append({
            "model": name,
            "accuracy": accuracy,
            "latency_ms_per_query": latency,
            "tokens_per_sec": throughput
        })
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Latency: {latency:.3f} ms/query")
        print(f"  Throughput: {throughput:.2f} tokens/sec")

    # --- Save results ---
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    csv_path = os.path.join(results_dir, 'conditional_compute_benchmark.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nResults saved to {csv_path}")

if __name__ == '__main__':
    # Need to import TransformerBlock for DenseEquivalent
    from toy_hydra import TransformerBlock
    conditional_compute_benchmark()
