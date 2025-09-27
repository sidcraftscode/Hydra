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
- `speedup_summary.py` — Aggregates benchmark outputs and renders speedup tables.
- `plot_results.py` — Plots figures from consolidated results.
- `logic_benchmark.py` — Logic composition benchmark (implication chains) comparing Hydra workspace ON/OFF vs Transformer.

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

- Logic composition benchmark (implication chains)
  - Generates chains like `A->B, B->C, C->D`, queries `A->?`, and measures accuracy vs proof length (2–5 steps).
  - Compares three models: Hydra workspace ON, Hydra workspace OFF, and Transformer baseline with similar parameter scale.
  - Results saved to `results/logic_composition_accuracy.csv`.
  
  ```bash
  python logic_benchmark.py
  ```

- Aggregate speedups and render a markdown table/JSON
  
  ```bash
  python speedup_summary.py
  ```

- Plot figures from `results/`
  
  ```bash
  python plot_results.py
  ```

Notes:
- Default toy config uses d=256, 8 blocks, attention every 4th block, MoE on even blocks, Top-2 routing.
- Scripts generate CSV/JSON artifacts such as `throughput_summary.csv`, `speedup_summary.json`, `train_losses.csv` in `results/`.
 - Logic benchmark trains on lengths 2–4 and evaluates on 2–5. Workspace memory is reset per batch for fairness. Expect Hydra workspace ON > Hydra workspace OFF > Transformer as proof length increases.

## Citation
If you use this code, please cite the paper:

Hydra: A 1.6B-Parameter State-Space Language Model with Sparse Attention, Mixture-of-Experts, and Memory

https://arxiv.org/abs/2508.15099
