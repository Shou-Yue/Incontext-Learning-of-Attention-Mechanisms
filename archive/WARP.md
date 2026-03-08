# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This is a replication of "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (Garg et al., NeurIPS 2022). The project trains a GPT-2 style Transformer to learn linear regression in-context without gradient updates.

Paper: https://arxiv.org/abs/2208.01066

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training

Run small test locally (50k steps for validation):
```bash
python src/training/train.py --config configs/test_local.json
# Or use convenience script:
./scripts/run_train.sh test_local
```

Run full training (500k steps):
```bash
python src/training/train.py --config configs/train_linear_20d.json
# Or:
./scripts/run_train.sh full
```

Run intermediate validation (50k steps):
```bash
python src/training/train.py --config configs/validate_50k.json
```

Training automatically resumes from latest checkpoint if interrupted.

### Evaluation

Evaluate trained model against all baseline methods:
```bash
python scripts/evaluate_all_methods.py \
  --checkpoint checkpoints/linear_20d_full/checkpoint_latest.pt \
  --n_dims 20 \
  --output results/
```

Specific evaluation scripts:
```bash
python scripts/evaluate_overparameterized.py    # Test overparameterized regime (n≥d)
python scripts/compare_overparameterized.py     # Compare methods in overparameterized regime
python scripts/evaluate_7d_15p.py               # Specific 7D/15-point configuration
```

### Visualization

Interactive plot with controls:
```bash
python scripts/interactive_plot.py results/all_methods_results.json
```

Training progress plots:
```bash
python scripts/plot_training.py checkpoints/linear_20d_full/training_log.jsonl
```

### Monitoring

Check training progress:
```bash
tail -f checkpoints/linear_20d_full/training_log.jsonl
tail -1 checkpoints/linear_20d_full/training_log.jsonl | python -m json.tool
```

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Architecture

### Core Design

The model uses an **interleaved sequence representation** for in-context learning:
- Input format: `[x1, y1, x2, y2, ..., xk, yk, x_query]`
- Each y value is padded with zeros to match input dimension
- The transformer predicts at y positions after seeing the corresponding x (causal masking)
- This allows the model to learn from (x,y) pairs without explicit gradient updates

### Model Components

**InContextTransformer** (`src/models/transformer.py`):
- GPT-2 architecture: 12 layers, 8 attention heads, 256 embedding dim (~22.4M parameters)
- Input projection: maps n_dims → n_embd 
- Transformer backbone: causal self-attention for in-context learning
- Output projection: maps n_embd → 1 (scalar prediction)
- Key method: `_combine(xs, ys)` interleaves inputs and outputs into sequence

**Data Pipeline** (`src/data/`):
- `samplers.py`: Gaussian sampler generates inputs from N(0, I_d)
- `tasks.py`: LinearRegressionTask samples weight vectors w ~ N(0, I_d) and computes y = w^T x
- `curriculum.py`: Gradually increases problem complexity during training

**Training Loop** (`src/training/train.py`):
- Curriculum learning: starts at 5D/11 points, reaches 20D/41 points by step 32k
- Auto-checkpointing: saves `checkpoint_latest.pt` periodically, plus milestone checkpoints every 50k steps
- Auto-resume: automatically continues from last checkpoint when restarted
- Logs to JSONL: `training_log.jsonl` with step, loss, curriculum state

**Baseline Methods** (`src/evaluation/baselines.py`):
- Least Squares: optimal closed-form solution for overparameterized regime (n≥d)
- Ridge Regression: regularized solution for underparameterized regime (n<d)
- 3-Nearest Neighbors: k-NN averaging
- Averaging: simple mean baseline

### Curriculum Learning

The curriculum gradually increases task complexity:
- **Dimensions**: 5 → 20 (increment by 1 every 2000 steps)
- **Context length**: 11 → 41 points (increment by 2 every 2000 steps)
- Reaches full complexity at step 32,000
- This allows the model to learn simpler patterns first before tackling full problem

### Evaluation Strategy

The evaluation scripts (`scripts/evaluate_*.py`) measure squared error across different context lengths (1-40 examples):
- Sample 1000 test prompts per context length
- For each prompt: sample task, generate training data, generate test point
- Compare transformer predictions against baselines
- Plot mean error ± std across methods (reproduces paper's Figure 2)

## Configuration Files

All configs are in `configs/*.json`:
- `test_local.json`: 50k steps, full model size (for quick validation)
- `train_linear_20d.json`: 500k steps, full model size (for complete replication)
- `validate_50k.json`: 50k steps, full model size (intermediate testing)

Key parameters:
- `n_dims`: Input/output dimension (20)
- `max_n_points`: Maximum sequence length (41 = up to 40 examples + 1 query)
- `n_embd/n_layer/n_head`: Transformer architecture (256/12/8)
- `batch_size`: Training batch size (64-256)
- `use_curriculum`: Enable/disable curriculum learning
- `resume`: Auto-resume from checkpoint (default: true)

## Development Practices

### When Modifying Training Code

- Test changes with `test_local` config first (faster iteration)
- Monitor GPU utilization with `nvidia-smi` - should be >80% during training
- Check curriculum is working: dims and points should increase in logs
- Training auto-resumes by default - set `"resume": false` in config to force restart

### When Adding New Baselines

- Implement prediction function in `src/evaluation/baselines.py`
- Follow signature: `method(xs_train, ys_train, xs_test) -> y_pred`
- Add to evaluation scripts in `scripts/evaluate_*.py`
- Update plot colors/labels in visualization code

### When Implementing New Function Classes

The current implementation focuses on linear functions. To add other function classes (sparse linear, decision trees, 2-layer NNs):
1. Create new task class in `src/data/tasks.py` (follow `LinearRegressionTask` template)
2. Task must implement `evaluate(xs) -> ys`
3. Update config with appropriate parameters for new task
4. Consider adjusting model architecture if input/output format changes

### File Organization

- `src/`: Core implementation (models, data, training, evaluation)
- `configs/`: Training configuration files (JSON)
- `scripts/`: Standalone scripts for training, evaluation, visualization
- `checkpoints/`: Model checkpoints and training logs (auto-generated)
- `results/`: Evaluation results and plots (auto-generated)

## Important Notes

- **Checkpoint format**: All checkpoints include model state, optimizer state, curriculum state, and config
- **Sequence representation**: The interleaved format is crucial - predictions happen at y positions (indices 1, 3, 5, ...) where the model has seen the corresponding x but not y yet
- **Underparameterized vs overparameterized**: When n_points < n_dims, use ridge regression for baselines; when n_points ≥ n_dims, use least squares
- **Training time**: Full 500k step training takes ~3-6 days on NVIDIA L4 (AWS g6.2xlarge)
