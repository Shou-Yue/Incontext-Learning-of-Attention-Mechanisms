# In-Context Learning of Attention Mechanisms

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
├── scripts/               # Training, evaluation, plots, sweeps
├── checkpoints/           # Saved model checkpoints
├── results/               # Plots and evaluation outputs
└── reference_repo/        # Original reference materials
```

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
./scripts/run_train.sh test_local
./scripts/run_train.sh full
```

## Evaluation (Transformer vs Baselines)

Reproduce the paper-style comparison across context lengths:
```bash
python scripts/evaluate_all_methods.py \
  --checkpoint checkpoints/linear_20d_full/checkpoint_latest.pt \
  --n_dims 20 \
  --output results/
```

Additional evaluations:
```bash
python scripts/evaluate_overparameterized.py
python scripts/compare_overparameterized.py
python scripts/evaluate_7d_15p.py
```

## Attention Variant Experiments

Softmax vs low‑rank (Linformer‑style) vs GD sweep:
```bash
python scripts/lowrank_softmax_sweep.py \
  --d 20 \
  --num_layers_list 1 2 4 8 \
  --num_train_tasks 10000 \
  --num_eval_tasks 1000 \
  --output_dir results/lowrank_sweep
```

Sanity check for low‑rank equivalence at full rank:
```bash
python scripts/verify_lowrank_equivalence.py
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

Plot LSA results:
```bash
python scripts/plot_lsa_multilayer.py
```

## Visualization

Training progress:
```bash
python scripts/plot_training.py checkpoints/linear_20d_full/training_log.jsonl
```

Interactive comparison plot:
```bash
python scripts/interactive_plot.py results/all_methods_results.json
```

## Outputs

Typical outputs are written under `results/` and `checkpoints/`:
- Evaluation JSON files and plots from scripts.
- Checkpoints and training logs from `src/training/train.py`.

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
