import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import os

# Toy Hydra Model
class HydraHead(nn.Module):
    def __init__(self, d_model):
        super(HydraHead, self).__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        return self.dense(context)

class HydraLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(HydraLayer, self).__init__()
        self.heads = nn.ModuleList([HydraHead(d_model) for _ in range(num_heads)])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, workspace):
        # Workspace attention
        if workspace is not None and len(workspace) > 0:
            ws_key = torch.stack([item[0] for item in workspace])
            ws_val = torch.stack([item[1] for item in workspace])
            head_outputs = [head(x, ws_key, ws_val) for head in self.heads]
            x = self.ln1(x + sum(head_outputs))
        
        # Self-attention (simplified)
        sa_output = self.heads[0](x, x, x) # Simplified self-attention for this toy model
        x = self.ln1(x + sa_output)

        # FFN
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        return x

class Hydra(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, use_workspace=True):
        super(Hydra, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([HydraLayer(d_model, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.use_workspace = use_workspace
        self.workspace = []

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, self.workspace if self.use_workspace else None)
        return self.fc_out(h)

    def add_to_workspace(self, key, value):
        if self.use_workspace:
            key_emb = self.embedding(key)
            val_emb = self.embedding(value)
            self.workspace.append((key_emb.squeeze(0), val_emb.squeeze(0)))
    
    def clear_workspace(self):
        self.workspace = []

# Transformer Baseline
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer_encoder(embedded)
        return self.fc_out(output)

# --- Data Generation ---
def generate_implication_chain(length):
    """Generates a chain of implications like A->B, B->C, ..."""
    vocab = list(range(1, 27)) # A-Z as integers
    chain_nodes = random.sample(vocab, length + 1)
    facts = []
    for i in range(length):
        facts.append((chain_nodes[i], chain_nodes[i+1]))
    query = chain_nodes[0]
    answer = chain_nodes[-1]
    return facts, query, answer

def create_vocab_and_mappings():
    """Creates a vocabulary and mappings for the logic symbols."""
    symbols = [chr(ord('A') + i) for i in range(26)] + ['->', '?']
    vocab = {s: i for i, s in enumerate(symbols)}
    inv_vocab = {i: s for s, i in vocab.items()}
    return vocab, inv_vocab

VOCAB, INV_VOCAB = create_vocab_and_mappings()
VOCAB_SIZE = len(VOCAB)

def sentence_to_ids(a, b, is_query=False):
    """Converts a fact (A->B) or query (A->?) to token IDs."""
    if is_query:
        return torch.tensor([VOCAB[INV_VOCAB[a]], VOCAB['->'], VOCAB['?']], dtype=torch.long).unsqueeze(0)
    return torch.tensor([VOCAB[INV_VOCAB[a]], VOCAB['->'], VOCAB[INV_VOCAB[b]]], dtype=torch.long).unsqueeze(0)

# --- Training and Evaluation ---
def train(model, facts, query, answer, optimizer, criterion):
    model.train()
    
    if isinstance(model, Hydra):
        model.clear_workspace()
        # For Hydra, we load facts into the workspace
        for fact_a, fact_b in facts:
            key = torch.tensor([VOCAB[INV_VOCAB[fact_a]]], dtype=torch.long)
            value = torch.tensor([VOCAB[INV_VOCAB[fact_b]]], dtype=torch.long)
            model.add_to_workspace(key, value)
        
        # Query the model
        input_seq = sentence_to_ids(query, None, is_query=True)
        target = torch.tensor([VOCAB[INV_VOCAB[answer]]], dtype=torch.long)

    elif isinstance(model, Transformer):
        # For Transformer, we concatenate facts and query
        fact_tensors = [sentence_to_ids(a, b) for a, b in facts]
        query_tensor = sentence_to_ids(query, None, is_query=True)
        input_seq = torch.cat(fact_tensors + [query_tensor], dim=1)
        # The target is the last element of the sequence
        target = torch.tensor([VOCAB[INV_VOCAB[answer]]], dtype=torch.long)

    optimizer.zero_grad()
    output = model(input_seq)
    
    # We only care about the last token's output for the answer
    loss = criterion(output[:, -1, :], target)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, facts, query, answer):
    model.eval()
    with torch.no_grad():
        if isinstance(model, Hydra):
            model.clear_workspace()
            for fact_a, fact_b in facts:
                key = torch.tensor([VOCAB[INV_VOCAB[fact_a]]], dtype=torch.long)
                value = torch.tensor([VOCAB[INV_VOCAB[fact_b]]], dtype=torch.long)
                model.add_to_workspace(key, value)
            input_seq = sentence_to_ids(query, None, is_query=True)
        
        elif isinstance(model, Transformer):
            fact_tensors = [sentence_to_ids(a, b) for a, b in facts]
            query_tensor = sentence_to_ids(query, None, is_query=True)
            input_seq = torch.cat(fact_tensors + [query_tensor], dim=1)

        output = model(input_seq)
        prediction = output[:, -1, :].argmax(dim=1)
        return prediction.item() == VOCAB[INV_VOCAB[answer]]

# --- Main Benchmark Loop ---
def run_benchmark():
    d_model = 64
    num_layers = 3
    num_heads = 4
    lr = 0.001
    epochs = 1000 # Increased epochs for better convergence
    
    models = {
        "Hydra (Workspace ON)": Hydra(VOCAB_SIZE, d_model, num_layers, num_heads, use_workspace=True),
        "Hydra (Workspace OFF)": Hydra(VOCAB_SIZE, d_model, num_layers, num_heads, use_workspace=False),
        "Transformer": Transformer(VOCAB_SIZE, d_model, num_heads, num_layers)
    }
    
    optimizers = {name: optim.Adam(model.parameters(), lr=lr) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    
    results = {name: [] for name in models.keys()}
    chain_lengths = range(2, 6)

    for length in chain_lengths:
        print(f"--- Testing Chain Length: {length} ---")
        for name, model in models.items():
            # Train the model on chains of the current length
            for epoch in range(epochs):
                facts, query, answer = generate_implication_chain(length)
                loss = train(model, facts, query, answer, optimizers[name], criterion)
                if epoch % 200 == 0:
                    print(f"  {name} - Epoch {epoch}, Loss: {loss:.4f}")

            # Evaluate the model
            correct = 0
            num_eval = 100
            for _ in range(num_eval):
                facts, query, answer = generate_implication_chain(length)
                if evaluate(model, facts, query, answer):
                    correct += 1
            accuracy = correct / num_eval
            results[name].append(accuracy)
            print(f"  {name} - Accuracy: {accuracy:.2f}\n")

    # Save results to CSV
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    csv_path = os.path.join(results_dir, 'logic_benchmark_accuracy.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'proof_length', 'accuracy'])
        for name, accuracies in results.items():
            for i, acc in enumerate(accuracies):
                writer.writerow([name, chain_lengths[i], acc])
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    run_benchmark()
