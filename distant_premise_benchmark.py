import torch
import torch.nn.functional as F
import time
import csv
import os
import random
from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters

# ------------------ Data Generation ------------------

def generate_distant_premise_data(seq_len=4000, premise_pos=2000, vocab_size=4000):
    """
    Generates a sequence with a premise and a query that requires the premise.
    Premise: "The color of the sky is [color]."
    Query: "What color is the sky?"
    """
    # Simple tokenization
    color_vocab = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    color = random.choice(color_vocab)
    
    # Let's use integers for simplicity, assuming a vocab mapping exists
    # For this toy example, we'll just use random integers, but the logic holds.
    # A real implementation would use a proper tokenizer.
    
    premise_tokens = [random.randint(100, vocab_size-1) for _ in range(5)] # "The color of the sky is"
    color_token = [color_vocab.index(color) + 1000] # Assign a unique-ish ID
    
    query_tokens = [random.randint(100, vocab_size-1) for _ in range(5)] # "What color is the sky?"
    
    sequence = torch.zeros(seq_len, dtype=torch.long)
    
    # Place premise
    premise_start = premise_pos
    sequence[premise_start:premise_start+len(premise_tokens)] = torch.tensor(premise_tokens)
    sequence[premise_start+len(premise_tokens)] = color_token[0]
    
    # Place query at the end
    query_start = seq_len - len(query_tokens) - 2 # leave space for question mark and answer
    sequence[query_start:query_start+len(query_tokens)] = torch.tensor(query_tokens)
    
    # Add a placeholder for where the model should predict
    sequence[query_start+len(query_tokens)] = 2 # '?' token
    
    return sequence.unsqueeze(0), color_token[0]

# ------------------ Training ------------------

def train(model, optimizer, criterion, data, device):
    """Performs a single training step."""
    model.train()
    
    sequence, target = data
    sequence, target = sequence.to(device), torch.tensor([target], device=device)
    
    optimizer.zero_grad()
    logits = model(sequence)
    
    # Loss is calculated on the token before the end of the sequence
    loss = criterion(logits[:, -2, :], target)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# ------------------ Benchmark ------------------

@torch.no_grad()
def run_benchmark(model, data, device):
    """Runs the model and returns accuracy, latency, and memory."""
    model.eval()
    
    sequence, target = data
    sequence = sequence.to(device)
    
    # Warmup
    for _ in range(2):
        _ = model(sequence)
        
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    t0 = time.time()
    logits = model(sequence)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    t1 = time.time()
    
    latency = (t1 - t0) * 1000 # ms
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if device.startswith('cuda') else 0.0
    
    # Get prediction for the last token
    pred = logits[0, -2, :].argmax().item() # predict for the token before last
    
    accuracy = 1.0 if pred == target else 0.0
    
    return accuracy, latency, peak_mem

# ------------------ Main ------------------

def distant_premise_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # --- Configurations ---
    base_cfg = HydraConfig(d=256, n_blocks=8, attn_every=2, vocab_size=5000, fast_ssm=True)
    
    # --- Model Variants ---
    models = {
        "transformer": BaselineTransformer(d=base_cfg.d, vocab_size=base_cfg.vocab_size, n_layers=base_cfg.n_blocks),
        "hydra_sparse_attn_on": ToyHydra(base_cfg),
        "hydra_sparse_attn_off": ToyHydra(HydraConfig(**{**base_cfg.__dict__, 'disable_attn': True})),
    }

    results = []
    
    seq_len = 4096
    premise_pos = 2000
    epochs = 100
    lr = 0.001

    for name, model in models.items():
        print(f"--- Benchmarking {name} ---")
        model.to(device)
        
        # --- Training Phase ---
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        print("Training...")
        for epoch in range(epochs):
            train_data = generate_distant_premise_data(seq_len=seq_len, premise_pos=premise_pos, vocab_size=base_cfg.vocab_size)
            loss = train(model, optimizer, criterion, train_data, device)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        # --- Evaluation Phase ---
        print("Evaluating...")
        # Run multiple times for stable results
        accuracies, latencies, memories = [], [], []
        for _ in range(5): # 5 runs
            eval_data = generate_distant_premise_data(seq_len=seq_len, premise_pos=premise_pos, vocab_size=base_cfg.vocab_size)
            accuracy, latency, memory = run_benchmark(model, eval_data, device)
            accuracies.append(accuracy)
            latencies.append(latency / seq_len)
            memories.append(memory)

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_latency = sum(latencies) / len(latencies)
        avg_memory = sum(memories) / len(memories)

        results.append({
            "model": name,
            "accuracy": avg_accuracy,
            "latency_ms_per_token": avg_latency,
            "peak_mem_MB": avg_memory,
            "seq_len": seq_len,
            "premise_pos": premise_pos
        })
        
        print(f"  Accuracy: {avg_accuracy:.2f}")
        print(f"  Latency per token: {avg_latency:.4f} ms")
        print(f"  Peak Memory: {avg_memory:.2f} MB")
        
        model.to('cpu') # free up memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # --- Save results ---
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    csv_path = os.path.join(results_dir, 'distant_premise_benchmark.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nResults saved to {csv_path}")

if __name__ == '__main__':
    distant_premise_benchmark()

