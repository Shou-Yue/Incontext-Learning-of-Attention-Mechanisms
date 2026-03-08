# In-Context Learning of Attention Mechanisms

Project website: [https://dhp003.github.io/dsc180b-website/](https://dhp003.github.io/dsc180b-website/)

This repo contains a clean, runnable set of experiments for in‑context linear regression. It combines:
- A GPT‑2–style transformer for in‑context learning (Garg et al., NeurIPS 2022).
- Mechanistic baselines and attention variants (softmax, low‑rank/Linformer‑style, kernelized linear attention).
- Linear Self‑Attention (LSA) models that approximate gradient descent (Oswald et al., NeurIPS 2023).

The core task is synthetic linear regression: sample weights `w ~ N(0, I)`, inputs `x ~ N(0, I)`, and labels `y = w^T x`.

## Project Structure

```
.
├── src/
│   ├── data/              # Samplers, tasks, curriculum
│   ├── models/
│   │   ├── transformer.py # GPT‑2 style transformer for ICL
│   │   ├── lsa.py         # Linear Self‑Attention models
│   │   └── attention_variants.py  # Softmax, low‑rank, kernel attention blocks
│   ├── training/
│   │   └── train.py       # Training loop for GPT‑2 ICL model
│   └── evaluation/        # Baselines and GD comparisons
├── configs/               # Training configs (JSON)
├── scripts/               # Active training, sweeps, plots, utilities
│   └── legacy/            # Older/unused scripts kept for reference
├── checkpoints/           # Saved model checkpoints
├── results/               # Plots and evaluation outputs
└── reference_repo/        # Original reference materials
```

## Results Layout (refactored)

We keep the **current run** directly under `results/` and archive everything else:

```
results/
├── experiments_002550_20260304/   # current run
│   ├── exp_steps_sweep/
│   ├── exp_layers_sweep/
│   ├── exp_context_sweep/
│   ├── exp_context_sweep_zero/
│   └── plots/
└── old-results/                   # archived runs + older artifacts
```

If you pull results from DSMLP, use `pull-and-plot.sh` to sync and regenerate plots locally.

## Setup

```bash
pip install -r requirements.txt
```

## Training (GPT‑2 ICL model)

Small/local run:
```bash
python src/training/train.py --config configs/test_local.json
```

Full run:
```bash
python src/training/train.py --config configs/train_linear_20d.json
```

Convenience wrapper:
```bash
./scripts/legacy/run_train.sh test_local
./scripts/legacy/run_train.sh full
```

## Evaluation (Transformer vs Baselines)

Reproduce the paper-style comparison across context lengths:
```bash
python scripts/legacy/evaluate_all_methods.py \
  --checkpoint checkpoints/linear_20d_full/checkpoint_latest.pt \
  --n_dims 20 \
  --output results/
```

Additional evaluations:
```bash
python scripts/legacy/evaluate_overparameterized.py
python scripts/legacy/compare_overparameterized.py
python scripts/legacy/evaluate_7d_15p.py
```

## Attention Variant Experiments

Softmax vs low‑rank (Linformer‑style) vs GD sweep:
```bash
python scripts/legacy/lowrank_softmax_sweep.py \
  --d 20 \
  --num_layers_list 1 2 4 8 \
  --num_train_tasks 10000 \
  --num_eval_tasks 1000 \
  --output_dir results/lowrank_sweep
```

Sanity check for low‑rank equivalence at full rank:
```bash
python scripts/legacy/verify_lowrank_equivalence.py
```

## Core Experiment Suite (4 Sweeps)

These are the main experiments we use to compare attention mechanisms. Each sweep outputs:
- `all_results.json` with MSE + cosine similarity vs GD
- Two plots (MSE + cosine)

### One‑command runner (recommended)
Creates a timestamped directory for every run:
`results/experiments_<HHMMSS_YYYYMMDD>/`
```bash
python scripts/run_all_experiments.py --device cuda
```
Plots go to `results/experiments_<...>/plots/`.

### Individual sweeps (manual)
You can still run the sweeps one‑by‑one. If you do, set `--output_dir` manually.

### 1) Training‑Steps Sweep (learnability)
Vary optimizer steps while holding layers and context length fixed.
```bash
python scripts/exp_steps_sweep.py \
  --d 20 \
  --n_points 41 \
  --num_layers 8 \
  --train_steps_list 0 1000 2000 5000 10000 20000 \
  --output_dir results/experiments_002550_20260304/exp_steps_sweep
```

### 2) Layer Sweep (layers ≈ GD steps)
Vary number of layers while holding training steps and context length fixed.
Uses a per‑layer step schedule (5k for 2/4/8, 10k for 16):
```bash
python scripts/exp_layers_sweep_stepsched.py \
  --d 20 \
  --n_points 41 \
  --num_layers_list 2 4 8 16 \
  --steps_shallow 5000 \
  --steps_deep 10000 \
  --deep_layer_threshold 16 \
  --output_dir results/experiments_002550_20260304/exp_layers_sweep
```

### 3) In‑Context Sweep (scaling with n)
Vary number of in‑context examples while holding layers and training steps fixed.
```bash
python scripts/exp_context_sweep.py \
  --d 20 \
  --n_points_list 5 10 20 40 80 120 160 200 \
  --num_layers 8 \
  --train_steps 5000 \
  --output_dir results/experiments_002550_20260304/exp_context_sweep
```

### 4) Zero‑Training Context Sweep (random init baseline)
Context sweep with **0 training** to show random‑init behavior.
```bash
python scripts/exp_context_sweep_zero_train_long.py \
  --d 20 \
  --n_points_list 5 10 20 40 80 120 160 200 \
  --num_layers 8 \
  --output_dir results/experiments_002550_20260304/exp_context_sweep_zero
```

### Interactive plots (recommended)
```bash
python scripts/plot_interactive_experiments.py \
  --steps_results results/experiments_002550_20260304/exp_steps_sweep/all_results.json \
  --layers_results results/experiments_002550_20260304/exp_layers_sweep/all_results.json \
  --context_results results/experiments_002550_20260304/exp_context_sweep/all_results.json \
  --context_zero_results results/experiments_002550_20260304/exp_context_sweep_zero/all_results.json \
  --output_dir results/experiments_002550_20260304/plots
```

## Pull results from DSMLP

Pull latest results and regenerate plots locally:
```bash
./pull-and-plot.sh latest auto --fresh
```

## LSA (Linear Self‑Attention) Experiments

Train and evaluate multi‑layer LSA models:
```bash
python scripts/lsa_gd_multilayer.py \
  --d 20 \
  --num_layers_list 1 2 4 8 16 32 \
  --num_epochs 30 \
  --device cuda
```

Legacy plotting utilities live under `scripts/legacy/`.

## Visualization

Training progress:
```bash
python scripts/legacy/plot_training.py checkpoints/linear_20d_full/training_log.jsonl
```

Interactive comparison plot:
```bash
python scripts/legacy/interactive_plot.py results/all_methods_results.json
```

## Outputs

Typical outputs are written under `results/` and `checkpoints/`:
- Evaluation JSON files and plots from scripts.
- Checkpoints and training logs from `src/training/train.py`.
- Core suite results in `results/experiments_002550_20260304/` (four sweeps + plots) or in
  `results/experiments_<HHMMSS_YYYYMMDD>/` when using `scripts/run_all_experiments.py`.

## Reproducibility

- Config-driven runs live in `configs/` (e.g., `configs/train_linear_20d.json`, `configs/test_local.json`, `configs/validate_50k.json`).
- Deterministic scripts for each experiment are under `scripts/` with CLI flags for hyperparameters and output directories.
- Plots and metrics are saved to `results/` for each run (JSON + PNG).
- Training checkpoints and logs are saved to `checkpoints/` for re‑evaluation.

## Notes

- Prompts are interleaved `[x1, y1, x2, y2, ..., xn, yn]` for transformer training.
- Low‑rank attention uses Linformer‑style sequence projection of K/V.
- LSA stacks correspond to multi‑step gradient descent behavior.

## References

- Garg et al., “What Can Transformers Learn In‑Context? A Case Study of Simple Function Classes” (NeurIPS 2022)
- Oswald et al., “Transformers Learn In‑Context by Gradient Descent” (NeurIPS 2023)
- x
