import csv, pathlib, json, math
from collections import defaultdict

RES_DIR = pathlib.Path('results')
LOSS_CSV = RES_DIR / 'train_losses.csv'
ALT_2000 = RES_DIR / 'train_losses_2000steps.csv'
OUT_JSON = RES_DIR / 'final_loss_summary.json'

# Prefer 2000-step file if present
loss_file = ALT_2000 if ALT_2000.exists() else LOSS_CSV

per_model = defaultdict(list)
with open(loss_file) as f:
    reader = csv.DictReader(f)
    for r in reader:
        per_model[r['model']].append((int(r['step']), float(r['loss'])))

summary = {}
for m, vals in per_model.items():
    vals = sorted(vals)
    final_step, final_loss = vals[-1]
    summary[m] = {
        'final_step': final_step,
        'final_loss': final_loss,
        'initial_logged_step': vals[0][0],
        'initial_logged_loss': vals[0][1],
        'loss_reduction_pct': 100.0 * (vals[0][1] - final_loss) / max(vals[0][1], 1e-9)
    }

# Relative parity gap vs transformer if both present
if 'transformer_base' in summary:
    base_loss = summary['transformer_base']['final_loss']
    for k,v in summary.items():
        v['delta_vs_transformer_pct'] = 100.0 * (v['final_loss'] - base_loss) / base_loss

with open(OUT_JSON,'w') as f:
    json.dump(summary, f, indent=2)

print('Wrote', OUT_JSON)
