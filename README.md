# In-Context Learning Replication

Replication and extension of:
- "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (Garg et al., NeurIPS 2022)
- "Transformers Learn In-Context by Gradient Descent" (Oswald et al., NeurIPS 2023)

This project investigates how transformers perform in-context learning on linear regression tasks and demonstrates that multi-layer Linear Self-Attention (LSA) models implement multi-step gradient descent.

## Project Structure

```
.
├── src/
│   ├── data/           # Data generation modules
│   ├── models/         # Transformer and LSA model architectures
│   │   ├── transformer.py  # GPT-2 style transformer for ICL
│   │   └── lsa.py          # Linear Self-Attention models
│   ├── training/       # Training loop and utilities
│   └── evaluation/     # Evaluation and baseline implementations
│       ├── baselines.py    # Least squares, k-NN, averaging
│       └── gd_baseline.py  # Gradient descent baselines for LSA
├── configs/            # Configuration files (JSON)
├── checkpoints/        # Model checkpoints
├── results/            # Evaluation results and plots
│   └── lsa_experiments/
│       └── final_results/  # Final LSA multi-layer results
├── archive/           # Old documentation and results
└── scripts/            # Training and evaluation scripts
    ├── lsa_gd_multilayer.py      # Multi-layer LSA experiments
    ├── plot_lsa_multilayer.py    # LSA visualization
    ├── aws_lsa_deploy.sh         # Deploy to AWS
    └── aws_lsa_retrieve.sh       # Retrieve results from AWS
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Run small test locally (1k steps)
python src/training/train.py --config configs/test_local.json

# Run full training (500k steps)
python src/training/train.py --config configs/train_linear_20d.json

# Or use convenience script
./scripts/run_train.sh test_local
./scripts/run_train.sh full
```

## Evaluation

```bash
# Evaluate transformer against all baseline methods
# Requires a trained model checkpoint
python scripts/evaluate_all_methods.py \
  --checkpoint checkpoints/linear_20d_full/checkpoint_latest.pt \
  --n_dims 20 \
  --output results/

# Other evaluation scripts
python scripts/evaluate_overparameterized.py  # Test overparameterized regime
python scripts/evaluate_7d_15p.py             # Specific configuration
```

## Visualization

```bash
# Interactive plot with adjustable smoothing and filtering
python scripts/interactive_plot.py results/all_methods_results.json

# Training progress plots
python scripts/plot_training.py checkpoints/linear_20d_full/training_log.jsonl
```

### Interactive Plot Controls

The interactive visualization includes:
- **Smoothing slider** (0-5): Apply Gaussian smoothing to curves
- **Show Error Bands**: Toggle confidence interval shading
- **X-range buttons**: Focus on underparameterized (n<d) or overparameterized (n≥d) regimes
- **Y-scale buttons**: Switch between linear and log scale

## Results

Evaluation outputs are saved in `results/`:
- `all_methods_results.json`: Performance metrics (mean, std) for all methods
- `all_methods_comparison.png`: Static comparison plot
- `training_progress.png`: Training loss over time

## Paper Reference

- Paper: https://arxiv.org/abs/2208.01066
- Original repo: https://github.com/dtsip/in-context-learning

---

## Part II: Multi-Layer LSA = Multi-Step Gradient Descent

**Based on:** Oswald et al. (2023) "Transformers Learn In-Context by Gradient Descent"

### Overview

We extend the single-layer LSA = GD result to demonstrate that stacking multiple Linear Self-Attention layers corresponds to performing multiple steps of gradient descent:
- **1 LSA layer** ≈ **1-step gradient descent**
- **L LSA layers** ≈ **L-step gradient descent**

### Quick Start

```bash
# Train and evaluate multi-layer LSA models
python scripts/lsa_gd_multilayer.py \
  --d 20 \
  --num_layers_list 1 2 4 8 16 32 \
  --sigma 0.0 \
  --num_epochs 30 \
  --device cuda

# Generate plots
python scripts/plot_lsa_multilayer.py
```

### Model Architecture

**LinearSelfAttentionBlock:**
- Linear Q, K, V projections (no softmax)
- Causal attention with masking
- Layer normalization (pre and post)
- Residual connections

**MultiLayerLSA:**
- Stacks L LSA blocks sequentially
- Processes interleaved tokens: `[x₁, y₁, x₂, y₂, ..., xₙ, yₙ, x_query]`
- Each layer refines the hidden representation
- Final prediction head extracts scalar output

**Key Implementation Details:**
- Xavier initialization with gain=0.01 for stability
- Gradient clipping (max_norm=1.0)
- Layer normalization essential for L ≥ 4
- Parameter count: ~1.8K (L=1) to ~31K (L=32)

### Training Setup

```bash
# Full training configuration
python scripts/lsa_gd_multilayer.py \
  --d 20                          # Input dimension
  --num_layers_list 1 2 4 8 16 32 # Layer depths to test
  --hidden_dim 20                 # Hidden dimension
  --num_train_tasks 10000         # Training tasks
  --num_eval_tasks 1000           # Evaluation tasks
  --batch_size 64                 # Batch size
  --num_epochs 30                 # Training epochs
  --lr 1e-3                       # Learning rate for LSA
  --eta 0.1                       # Learning rate for GD baseline
  --sigma 0.0                     # Label noise (set to 0.2 for noisy)
  --device cuda                   # Use GPU
```

**Training Details:**
- Tasks: Linear regression with d=20, n=41 points (overparameterized)
- Data: x ~ N(0, I_d), w ~ N(0, I_d), y = w^T x
- Optimizer: Adam
- Loss: Mean Squared Error

### Evaluation Metrics

1. **Prediction Accuracy (MSE)**
   - How error decreases with depth
   - Compare LSA vs. L-step gradient descent

2. **Update Alignment (Cosine Similarity)**
   - Alignment between LSA and GD weight updates
   - Extracted via least-squares: w_LSA = (X^T X)^{-1} X^T ŷ
   - Values close to 1.0 indicate GD-like behavior

### Experimental Results

**Summary Table** (d=20, n=41, 30 epochs, σ=0.0):

| Layers | MSE LSA | MSE GD | Cosine Sim | Status |
|--------|---------|--------|------------|--------|
| 1      | 9.32    | 16.44  | 0.873      | LSA better |
| 2      | 6.53    | 13.22  | 0.900      | LSA better |
| 4      | 5.94    | 9.96   | 0.887      | LSA better |
| 8      | 5.15    | 6.31   | 0.905      | LSA better |
| 16     | 4.98    | 3.47   | 0.909      | GD better |
| 32     | 3.96    | 1.38   | 0.916      | GD better |

**Key Findings:**

✅ **High cosine similarity (>0.87)** across all depths confirms LSA implements GD-like updates

✅ **Crossover at L=16:** LSA outperforms GD at shallow depths (L≤8), but GD converges better at deep iterations (L≥16)

✅ **Mechanism preserved:** Despite MSE differences, directional alignment remains strong, indicating LSA learns the correct update rule

✅ **Diminishing returns:** Both methods show smaller improvements at greater depth

### AWS Deployment

For GPU training, use the automated AWS workflow:

```bash
# 1. Deploy to AWS (automatically packages and uploads code)
./scripts/aws_lsa_deploy.sh

# 2. SSH into AWS instance and run experiment
ssh -i ~/.ssh/your-key.pem ec2-user@your-instance-ip
cd ~/lsa-icl
source venv/bin/activate
python scripts/lsa_gd_multilayer.py \
  --d 20 --num_layers_list 1 2 4 8 16 32 \
  --sigma 0.0 --num_epochs 30 --device cuda

# 3. Retrieve results back to local machine
./scripts/aws_lsa_retrieve.sh
```

**Note:** AWS credentials are configured in the deployment scripts. See `AWS_WORKFLOW.md` for setup instructions.

### Visualization

```bash
# Generate all plots
python scripts/plot_lsa_multilayer.py

# Or specify custom paths
python scripts/plot_lsa_multilayer.py \
  --results_file ./results/lsa_experiments/final_results/all_results.json \
  --output_dir ./results/lsa_experiments/final_results
```

**Generated plots:**
- `lsa_vs_gd_mse.png`: MSE comparison across depths
- `lsa_vs_gd_cosine.png`: Cosine similarity across depths
- `lsa_vs_gd_combined.png`: Both metrics side-by-side
- `results_table.csv`: Numerical results

### Technical Notes

**Why Layer Normalization?**
Without LayerNorm, models with L≥4 produce NaN losses. Pre-LN and post-LN stabilize deep networks.

**Why Small Initialization?**
Gain=0.01 prevents gradient explosion. Standard gain=1.0 causes training failures.

**Weight Extraction Method:**
We evaluate LSA on random test points and solve w = (X^T X)^{-1} X^T ŷ to recover the implicit linear function.

**GD Crossover Explanation:**
At L≥16, GD's analytical convergence outpaces LSA's learned optimization. LSA approximation errors compound across many layers, while GD remains exact.

### Additional Documentation

- **Archived docs:** See `archive/` for detailed implementation notes, AWS workflow, and previous results
- **Paper:** [Oswald et al., NeurIPS 2023](https://arxiv.org/abs/2212.07677)

---

## Implementation Status

### Part I: In-Context Learning
- [x] Linear functions (20D) - 500k steps
- [x] Baseline methods (Least Squares, k-NN, Averaging)
- [x] Interactive visualization
- [ ] Distribution shift experiments
- [ ] Sparse linear functions
- [ ] Decision trees
- [ ] 2-layer neural networks

### Part II: LSA = Gradient Descent
- [x] Single-layer LSA = 1-step GD
- [x] Multi-layer LSA (1, 2, 4, 8, 16, 32 layers)
- [x] MSE and cosine similarity evaluation
- [x] AWS deployment automation
- [x] Visualization and analysis
- [ ] 64+ layer experiments
- [ ] Noise robustness analysis (σ > 0)
