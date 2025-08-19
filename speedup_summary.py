import csv, pathlib, json, statistics as stats
from collections import defaultdict

RES_DIR = pathlib.Path('results')
TH_FILE = RES_DIR / 'throughput_summary.csv'
OUT_MD = RES_DIR / 'speedup_table.md'
OUT_JSON = RES_DIR / 'speedup_summary.json'

PRIMARY_BASE = 'transformer_base'
PRIMARY_HYDRA = 'hydra_base'
PARITY = 'hydra_parity'
ABLATE_NO_MOE = 'hydra_no_moe'
ABLATE_NO_ATTN = 'hydra_no_attn'

rows = []
with open(TH_FILE) as f:
    reader = csv.DictReader(f)
    for r in reader:
        r['seq_len'] = int(r['seq_len'])
        r['toks_per_s'] = float(r['toks_per_s'])
        r['ms_per_tok'] = float(r['ms_per_tok'])
        r['peak_mem_MB'] = float(r['peak_mem_MB'])
        rows.append(r)

# Group by model
by_model = defaultdict(dict)
for r in rows:
    by_model[r['model']][r['seq_len']] = r

# Determine common lengths
lengths = sorted(set.intersection(*[set(m.keys()) for m in by_model.values() if m]))

speedup_records = []
for L in lengths:
    if PRIMARY_BASE in by_model and PRIMARY_HYDRA in by_model:
        base = by_model[PRIMARY_BASE][L]['toks_per_s']
        hydra = by_model[PRIMARY_HYDRA][L]['toks_per_s']
        speed = hydra / base if base>0 else None
    else:
        speed = None
    rec = {
        'seq_len': L,
        'transformer_toks_s': by_model.get(PRIMARY_BASE, {}).get(L, {}).get('toks_per_s'),
        'hydra_base_toks_s': by_model.get(PRIMARY_HYDRA, {}).get(L, {}).get('toks_per_s'),
        'speedup_hydra_base_vs_transformer': speed,
        'hydra_parity_toks_s': by_model.get(PARITY, {}).get(L, {}).get('toks_per_s'),
        'peak_mem_transformer_MB': by_model.get(PRIMARY_BASE, {}).get(L, {}).get('peak_mem_MB'),
        'peak_mem_hydra_MB': by_model.get(PRIMARY_HYDRA, {}).get(L, {}).get('peak_mem_MB'),
    }
    speedup_records.append(rec)

# Ablation deltas at largest shared length
largest = max(lengths) if lengths else None
ablations = {}
if largest is not None:
    base_t = by_model.get(PRIMARY_BASE, {}).get(largest, {}).get('toks_per_s')
    hydra_t = by_model.get(PRIMARY_HYDRA, {}).get(largest, {}).get('toks_per_s')
    no_moe_t = by_model.get(ABLATE_NO_MOE, {}).get(largest, {}).get('toks_per_s')
    no_attn_t = by_model.get(ABLATE_NO_ATTN, {}).get(largest, {}).get('toks_per_s')
    ablations = {
        'length': largest,
        'hydra_base_toks_s': hydra_t,
        'no_moe_toks_s': no_moe_t,
        'no_attn_toks_s': no_attn_t,
        'delta_no_moe_pct': (no_moe_t/hydra_t - 1)*100 if hydra_t and no_moe_t else None,
        'delta_no_attn_pct': (no_attn_t/hydra_t - 1)*100 if hydra_t and no_attn_t else None,
    }

# Write JSON summary
with open(OUT_JSON,'w') as f:
    json.dump({'speedups': speedup_records, 'ablations_at_max_len': ablations}, f, indent=2)

# Attempt to append 16K row from long_context_stats.json (long context benchmark)
LONG_JSON = RES_DIR / 'long_context_stats.json'
try:
    with open(LONG_JSON) as f:
        long_stats = json.load(f)
    have_16k_already = any(rec['seq_len'] == 16384 for rec in speedup_records)
    if not have_16k_already:
        hydra_long = long_stats.get('hydra_base', {}).get('16384', {})
        trans_long = long_stats.get('transformer', {}).get('16384', {})
        if hydra_long and trans_long:
            h_tps = hydra_long.get('toks_s_mean')
            t_tps = trans_long.get('toks_s_mean')
            speed = h_tps / t_tps if (h_tps and t_tps) else None
            speedup_records.append({
                'seq_len': 16384,
                'transformer_toks_s': t_tps,
                'hydra_base_toks_s': h_tps,
                'speedup_hydra_base_vs_transformer': speed,
                'hydra_parity_toks_s': None,  # not measured at 16K in fairness benchmark
                'peak_mem_transformer_MB': trans_long.get('peak_mem_MB'),
                'peak_mem_hydra_MB': hydra_long.get('peak_mem_MB'),
            })
            # keep rows sorted
            speedup_records = sorted(speedup_records, key=lambda r: r['seq_len'])
except FileNotFoundError:
    pass

# Markdown table
def fmt(x, prec=2):
    if x is None: return '-' 
    return f"{x:.{prec}f}"

lines = []
lines.append('# Speedup Summary')
lines.append('')
lines.append('| Seq Len | Transformer toks/s | Hydra base toks/s | Speedup | Parity toks/s | PeakMem T (MB) | PeakMem H (MB) |')
lines.append('|--------:|--------------------:|-------------------:|--------:|---------------:|---------------:|---------------:|')
for rec in speedup_records:
    lines.append(f"| {rec['seq_len']:7d} | {fmt(rec['transformer_toks_s'],0)} | {fmt(rec['hydra_base_toks_s'],0)} | {fmt(rec['speedup_hydra_base_vs_transformer'],2)} | {fmt(rec['hydra_parity_toks_s'],0)} | {fmt(rec['peak_mem_transformer_MB'],0)} | {fmt(rec['peak_mem_hydra_MB'],0)} |")
lines.append('')
if ablations:
    lines.append(f"Ablations at length {largest}:")
    lines.append('')
    lines.append('| Variant | toks/s | % vs Hydra base |')
    lines.append('|---------|-------:|----------------:|')
    lines.append(f"| hydra_base | {fmt(ablations['hydra_base_toks_s'],0)} | 0.00% |")
    if ablations.get('no_moe_toks_s'):
        lines.append(f"| hydra_no_moe | {fmt(ablations['no_moe_toks_s'],0)} | {fmt(ablations['delta_no_moe_pct'],2)}% |")
    if ablations.get('no_attn_toks_s'):
        lines.append(f"| hydra_no_attn | {fmt(ablations['no_attn_toks_s'],0)} | {fmt(ablations['delta_no_attn_pct'],2)}% |")

with open(OUT_MD,'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Wrote', OUT_JSON, 'and', OUT_MD)
