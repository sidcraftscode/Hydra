# Hydra

This folder contains the toy-scale prototype code accompanying the Hydra paper:

Hydra: A 1.6B-Parameter State-Space Language Model with Sparse Attention, Mixture-of-Experts, and Memory

Paper: https://arxiv.org/abs/2508.15099

Important: These scripts are illustrative and operate at small scale on synthetic data to validate integration and scaling trends. They are not a full 1.6B model training pipeline.

## Contents
- `toy_hydra.py` — Minimal Hydra model (SSM + sparse attention + MoE) used in benchmarks.
- `ssm_kernels.py` — Placeholder selective-scan SSM; the fast surrogate lives in `toy_hydra.py`.
- `workspace_memory.py` — Latent workspace memory (read/write) toy implementation.
- `pkm_memory.py` — Product-Key Memory (PKM) toy layer and helpers.
- `fairness_benchmark.py` — Training loop and short-context throughput benchmark on synthetic tasks.
- `run_long_context.py` — Long-context throughput/memory runs (1k–16k tokens).
- `benchmark_multihop_lookup.py` — Multi-hop lookup accuracy vs hops (Hydra PKM ON vs OFF vs Transformer).
- `benchmark_logic_composition.py` — Logic composition accuracy vs proof length (Hydra workspace ON vs OFF vs Transformer).
- `speedup_summary.py` — Aggregates benchmark outputs and renders speedup tables.
- `plot_results.py` — Plots figures from consolidated results.

Outputs are written to a `results/` folder (see paper’s Reproducibility section for exact filenames).

## Requirements
- Python 3.11
- PyTorch ≥ 2.1
- NumPy
- Matplotlib (for plotting)

GPU is optional; CPU runs are slower. CUDA is only needed for the long-context throughput experiments reported in the paper.

## Quickstart
Create an environment and install dependencies, then run any of the scripts below from this folder.

Examples:

- Run benchmark + training (default)
  
  ```bash
  python fairness_benchmark.py
  ```

- Long-context benchmark (1k–16k)
  
  ```bash
  python run_long_context.py
  ```

- Aggregate speedups and render a markdown table/JSON
  
  ```bash
  python speedup_summary.py
  ```

- Plot figures from `results/`
  
  ```bash
  python plot_results.py
  ```

### New reasoning benchmarks

1) Multi-hop lookup (2–6 hops)

- Data: Create synthetic KB chains E1→E2, E2→E3, … with many distractor facts.
- Format: Facts list + `<scratch>` steps + query token; model predicts the final entity.
- Run: Hydra (PKM ON vs PKM OFF) vs Transformer under identical training budgets.
- Output: Accuracy vs hops saved to `results/multihop_lookup.{csv,json}`.

Run:

```bash
python benchmark_multihop_lookup.py
```

2) Logic composition (implication chains / simple propositional logic)

- Data: Chains like A→B, B→C, C→D with distractors; query A→?
- Run: Hydra workspace ON vs OFF vs Transformer with equal budgets.
- Output: Accuracy vs proof length (2–5 steps) saved to `results/logic_composition.{csv,json}`.

Run:

```bash
python benchmark_logic_composition.py
```

Notes:
- Default toy config uses d=256, 8 blocks, attention every 4th block, MoE on even blocks, Top-2 routing.
- Scripts generate CSV/JSON artifacts such as `throughput_summary.csv`, `speedup_summary.json`, `train_losses.csv` in `results/`.

## Citation
If you use this code, please cite the paper:

Hydra: A 1.6B-Parameter State-Space Language Model with Sparse Attention, Mixture-of-Experts, and Memory

https://arxiv.org/abs/2508.15099
