import json, csv, math
import pathlib
from collections import defaultdict
import matplotlib.pyplot as plt

RES_DIR = pathlib.Path('results')

# Load throughput summary
try:
    throughput_rows = []
    with open(RES_DIR / 'throughput_summary.csv') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['seq_len'] = int(r['seq_len']); r['toks_per_s'] = float(r['toks_per_s']); r['ms_per_tok'] = float(r['ms_per_tok']); r['peak_mem_MB'] = float(r['peak_mem_MB'])
            throughput_rows.append(r)

    # Group
    group = defaultdict(list)
    for r in throughput_rows:
        group[r['model']].append(r)
    for k in group:
        group[k] = sorted(group[k], key=lambda x: x['seq_len'])

    # Plot throughput vs length
    plt.figure()
    for model, rows in group.items():
        plt.plot([r['seq_len'] for r in rows], [r['toks_per_s'] for r in rows], marker='o', label=model)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Sequence length')
    plt.ylabel('Tokens / s')
    plt.title('Throughput vs Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / 'fig_throughput.png', dpi=150)

    # ms per token
    plt.figure()
    for model, rows in group.items():
        plt.plot([r['seq_len'] for r in rows], [r['ms_per_tok'] for r in rows], marker='o', label=model)
    plt.xscale('log', base=2)
    plt.xlabel('Sequence length')
    plt.ylabel('ms / token')
    plt.title('Latency per token')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / 'fig_latency.png', dpi=150)

    # Peak memory
    plt.figure()
    for model, rows in group.items():
        plt.plot([r['seq_len'] for r in rows], [r['peak_mem_MB'] for r in rows], marker='o', label=model)
    plt.xscale('log', base=2)
    plt.xlabel('Sequence length')
    plt.ylabel('Peak memory (MB)')
    plt.title('Peak Memory vs Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / 'fig_memory.png', dpi=150)

    # Speedup Hydra base vs transformer_base
    base_rows = {r['seq_len']: r for r in group.get('transformer_base', [])}
    hydra_rows = {r['seq_len']: r for r in group.get('hydra_base', [])}
    common = sorted(set(base_rows) & set(hydra_rows))
    if common:
        plt.figure()
        plt.plot(common, [hydra_rows[L]['toks_per_s']/base_rows[L]['toks_per_s'] for L in common], marker='o')
        plt.xscale('log', base=2)
        plt.xlabel('Sequence length')
        plt.ylabel('Speedup (Hydra base / Transformer)')
        plt.title('Hydra Speedup vs Length')
        plt.tight_layout()
        plt.savefig(RES_DIR / 'fig_speedup.png', dpi=150)
except FileNotFoundError:
    pass

# Training losses
loss_log = defaultdict(list)
try:
    with open(RES_DIR / 'train_losses.csv') as f:
        reader = csv.DictReader(f)
        for r in reader:
            loss_log[r['model']].append((int(r['step']), float(r['loss'])))
    plt.figure()
    for m, vals in loss_log.items():
        vals = sorted(vals)
        plt.plot([v[0] for v in vals], [v[1] for v in vals], marker='o', label=m)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / 'fig_train_loss.png', dpi=150)
except FileNotFoundError:
    pass

# PKM benchmark results
try:
    pkm_results = []
    with open(RES_DIR / 'pkm_benchmark_results.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['accuracy'] = float(row['accuracy'])
            row['latency_ms'] = float(row['latency_ms'])
            row['beta_activation'] = float(row['beta_activation'])
            pkm_results.append(row)

    if pkm_results:
        # Bar chart for beta activation
        plt.figure()
        scenarios = [r['scenario'] for r in pkm_results]
        betas = [r['beta_activation'] for r in pkm_results]
        plt.bar(scenarios, betas, color=['blue', 'orange'])
        plt.ylabel('Beta Activation')
        plt.title('PKM Beta Activation by Scenario')
        plt.tight_layout()
        plt.savefig(RES_DIR / 'fig_pkm_beta_activation.png', dpi=150)

except FileNotFoundError:
    pass

# Distant premise benchmark results
try:
    distant_results = []
    with open(RES_DIR / 'distant_premise_benchmark.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['accuracy'] = float(row['accuracy'])
            row['latency_ms_per_token'] = float(row['latency_ms_per_token'])
            row['peak_mem_MB'] = float(row['peak_mem_MB'])
            distant_results.append(row)

    if distant_results:
        models = [r['model'] for r in distant_results]
        accuracy = [r['accuracy'] for r in distant_results]
        latency = [r['latency_ms_per_token'] for r in distant_results]
        memory = [r['peak_mem_MB'] for r in distant_results]

        # Bar chart for accuracy
        plt.figure()
        plt.bar(models, accuracy)
        plt.ylabel('Accuracy (Exact Match)')
        plt.title('Distant Premise Reasoning Accuracy')
        plt.tight_layout()
        plt.savefig(RES_DIR / 'fig_distant_accuracy.png', dpi=150)

        # Bar chart for latency
        plt.figure()
        plt.bar(models, latency)
        plt.ylabel('Latency (ms/token)')
        plt.title('Distant Premise Reasoning Latency')
        plt.tight_layout()
        plt.savefig(RES_DIR / 'fig_distant_latency.png', dpi=150)
        
        # Bar chart for memory
        plt.figure()
        plt.bar(models, memory)
        plt.ylabel('Peak Memory (MB)')
        plt.title('Distant Premise Reasoning Memory')
        plt.tight_layout()
        plt.savefig(RES_DIR / 'fig_distant_memory.png', dpi=150)

except FileNotFoundError:
    pass

# Conditional compute benchmark results
try:
    conditional_results = []
    with open(RES_DIR / 'conditional_compute_benchmark.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['accuracy'] = float(row['accuracy'])
            row['latency_ms_per_query'] = float(row['latency_ms_per_query'])
            row['tokens_per_sec'] = float(row['tokens_per_sec'])
            conditional_results.append(row)

    if conditional_results:
        # Bar chart for accuracy
        plt.figure()
        models = [r['model'] for r in conditional_results]
        accuracy = [r['accuracy'] for r in conditional_results]
        plt.bar(models, accuracy)
        plt.ylabel('Accuracy (Exact Match)')
        plt.title('Conditional Compute: Accuracy')
        plt.tight_layout()
        plt.savefig(RES_DIR / 'fig_conditional_compute_accuracy.png', dpi=150)

except FileNotFoundError:
    pass

# Logic benchmark results
try:
    logic_results = defaultdict(list)
    with open(RES_DIR / 'logic_benchmark_accuracy.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            logic_results[row['model']].append((int(row['proof_length']), float(row['accuracy'])))

    if logic_results:
        plt.figure(figsize=(10, 6))
        for model, values in logic_results.items():
            values.sort()
            proof_lengths = [v[0] for v in values]
            accuracies = [v[1] for v in values]
            plt.plot(proof_lengths, accuracies, marker='o', linestyle='-', label=model)
        
        plt.title('Logic Composition Benchmark: Accuracy vs. Proof Length')
        plt.xlabel('Proof Length (Number of Implications)')
        plt.ylabel('Accuracy')
        # Use a sorted list of unique proof lengths for xticks
        all_proof_lengths = sorted(list(set(pl for values in logic_results.values() for pl, acc in values)))
        if all_proof_lengths:
            plt.xticks(all_proof_lengths)
        plt.ylim(0, 1.1)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RES_DIR / 'fig_logic_benchmark.png', dpi=150)

except FileNotFoundError:
    pass

print('Saved figures to results/*.png')
