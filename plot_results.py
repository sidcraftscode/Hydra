import json, csv, math
import pathlib
from collections import defaultdict
import matplotlib.pyplot as plt

RES_DIR = pathlib.Path('results')

# Load throughput summary
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
plt.figure()
plt.plot(common, [hydra_rows[L]['toks_per_s']/base_rows[L]['toks_per_s'] for L in common], marker='o')
plt.xscale('log', base=2)
plt.xlabel('Sequence length')
plt.ylabel('Speedup (Hydra base / Transformer)')
plt.title('Hydra Speedup vs Length')
plt.tight_layout()
plt.savefig(RES_DIR / 'fig_speedup.png', dpi=150)

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

print('Saved figures to results/*.png')
