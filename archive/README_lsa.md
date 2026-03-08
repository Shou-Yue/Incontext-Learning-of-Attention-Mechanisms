# Linear Self-Attention (LSA) = Gradient Descent Experiments

This directory contains experiments demonstrating that multi-layer Linear Self-Attention (LSA) models approximate multi-step gradient descent on linear regression tasks.

Based on: **Oswald et al. (2023): "Transformers Learn In-Context by Gradient Descent"**

## Overview

The key insight is that:
- A **single LSA layer** learns to approximate **one-step gradient descent**
- **Multiple stacked LSA layers** approximate **multi-step gradient descent (GD++)**

This implementation tests this hypothesis by:
1. Training multi-layer LSA models with `num_layers ∈ {1, 2, 4}`
2. Comparing their predictions to T-step gradient descent baselines
3. Measuring alignment via MSE and cosine similarity of weight updates

## Model Architecture

### LinearSelfAttentionBlock

A single LSA layer that processes tokens of the form `[x_i; y_i]`:

```python
class LinearSelfAttentionBlock(nn.Module):
    - Uses Q, K, V projections (no softmax!)
    - Linear attention: A = Q K^T
    - Causal masking for autoregressive prediction
    - Residual connections
```

### MultiLayerLSA

Stacks multiple LSA blocks sequentially:

```python
model = MultiLayerLSA(
    d=20,              # input dimension
    num_layers=4,      # number of LSA layers
    hidden_dim=20      # hidden dimension (default: d)
)
```

When `num_layers=1`, this reduces to the single-layer case.

## Gradient Descent Baselines

The baselines implement T-step gradient descent from `W_0 = 0`:

```
W_{t+1} = W_t - eta * (1/N) * sum_i (W_t x_i - y_i) x_i^T
```

Functions:
- `gd_one_step()`: One step of GD
- `gd_t_steps(T)`: T steps of GD

## Running Experiments

### Basic Usage

Train and evaluate LSA models for layers {1, 2, 4}:

```bash
python scripts/lsa_gd_multilayer.py --d 20 --num_layers_list 1 2 4 --sigma 0.0
```

### Full Options

```bash
python scripts/lsa_gd_multilayer.py \
  --d 20 \                          # Input dimension
  --num_layers_list 1 2 4 \         # List of layer depths to test
  --hidden_dim 20 \                 # Hidden dimension (default: d)
  --num_train_tasks 10000 \         # Number of training tasks
  --num_eval_tasks 1000 \           # Number of evaluation tasks
  --batch_size 64 \                 # Training batch size
  --num_epochs 10 \                 # Number of epochs
  --lr 1e-3 \                       # Learning rate for LSA training
  --eta 0.1 \                       # Learning rate for GD baseline
  --sigma 0.0 \                     # Label noise level (std dev)
  --output_dir ./results/lsa_multilayer \
  --device auto                     # Device (auto, cpu, cuda)
```

### With Label Noise

To test robustness under noisy labels:

```bash
python scripts/lsa_gd_multilayer.py --d 20 --num_layers_list 1 2 4 --sigma 0.2
```

## Generating Plots

After running experiments, generate visualizations:

```bash
python scripts/plot_lsa_multilayer.py \
  --results_file ./results/lsa_multilayer/all_results.json \
  --output_dir ./results/lsa_multilayer
```

This creates:
- `lsa_vs_gd_mse.png`: MSE comparison between LSA and GD
- `lsa_vs_gd_cosine.png`: Cosine similarity between LSA and GD weight updates
- `lsa_vs_gd_combined.png`: Combined plot with both metrics
- `results_table.csv`: Numerical results in CSV format

## Task Generation

Linear regression tasks are generated as follows:

```python
# Sample true weight vector: w ~ N(0, I_d)
w_true = torch.randn(d)

# Sample inputs: x_i ~ N(0, I_d)
xs = torch.randn(n_points, d)

# Compute labels: y_i = w^T x_i + noise
ys = xs @ w_true + sigma * torch.randn(n_points)
```

Default: `n_points = 2d + 1` (overparameterized regime)

## Evaluation Metrics

For each `num_layers`:

1. **MSE (Mean Squared Error)**
   - `MSE_LSA`: Error of LSA predictions
   - `MSE_GD`: Error of T-step GD predictions
   - Lower is better

2. **Cosine Similarity**
   - Alignment between LSA and GD weight updates
   - `cos_sim = <w_LSA, w_GD> / (||w_LSA|| ||w_GD||)`
   - Higher is better (1.0 = perfect alignment)

## Expected Results

If LSA successfully mimics multi-step GD, we expect:

1. **Low MSE**: Both LSA and GD should achieve low test error
2. **Decreasing MSE with depth**: More layers → better performance
3. **High cosine similarity**: LSA weight updates align with GD (cos_sim > 0.9)
4. **Similarity maintained across depths**: Alignment holds for all `num_layers`

## File Structure

```
src/
  models/
    lsa.py                      # LSA model implementations
  evaluation/
    gd_baseline.py              # Gradient descent baselines

scripts/
  lsa_gd_multilayer.py          # Main experiment driver
  plot_lsa_multilayer.py        # Plotting script

results/
  lsa_multilayer/               # Experiment outputs
    all_results.json            # Numerical results
    lsa_1layer_checkpoint.pt    # Model checkpoints
    lsa_2layer_checkpoint.pt
    lsa_4layer_checkpoint.pt
    lsa_vs_gd_mse.png          # Plots
    lsa_vs_gd_cosine.png
    lsa_vs_gd_combined.png
    results_table.csv
```

## Implementation Notes

### Data Distribution
- Inputs: `x ~ N(0, I_d)`
- Weights: `w ~ N(0, I_d)`
- Labels: `y = w^T x + epsilon`, where `epsilon ~ N(0, sigma^2)`

### Training Details
- Optimizer: Adam with learning rate `1e-3`
- Loss: Mean Squared Error
- Tasks per epoch: `num_train_tasks / batch_size`
- Default: 10,000 tasks over 10 epochs

### Hyperparameters
- **d**: Input dimension (default: 20)
- **n**: Number of training points = 2d + 1 (default: 41)
- **eta**: GD learning rate (default: 0.1)
- **sigma**: Label noise std dev (default: 0.0)

### Causal Masking
LSA uses causal attention, allowing each token to only attend to previous tokens. This ensures:
- Token at position `i` can see `{x_1, y_1, ..., x_i, y_i}`
- But NOT future tokens `{x_{i+1}, y_{i+1}, ...}`

This mimics the sequential nature of gradient descent.

## Example Output

```
============================================================
Experiment: num_layers = 1
============================================================
Training 1-layer LSA model...
  d=20, n=41, tasks=10000, batch_size=64
  epochs=10, lr=0.001, sigma=0.0
Epoch 1 avg loss: 0.234567
...
Epoch 10 avg loss: 0.012345

Evaluating 1-layer LSA model...
Results for 1-layer LSA:
  MSE LSA: 0.015234 ± 0.003456
  MSE GD:  0.014987 ± 0.003214
  Cosine similarity: 0.945678 ± 0.012345

============================================================
SUMMARY TABLE
============================================================
Layers     MSE LSA              MSE GD               Cosine Sim
----------------------------------------------------------------------
1          0.015234 ± 0.0034    0.014987 ± 0.0032    0.945678 ± 0.0123
2          0.008765 ± 0.0021    0.008543 ± 0.0019    0.963421 ± 0.0089
4          0.004321 ± 0.0012    0.004198 ± 0.0011    0.978654 ± 0.0067
============================================================
```

## Troubleshooting

### Low cosine similarity
- Increase training epochs
- Adjust learning rate for LSA
- Try different hidden dimension

### High MSE
- Check that `n = 2d + 1` for overparameterized regime
- Verify GD learning rate (eta) is appropriate
- Increase training tasks

### GPU memory issues
- Reduce batch size
- Reduce d (input dimension)
- Use smaller num_layers

## References

1. **Oswald et al. (2023)**: "Transformers Learn In-Context by Gradient Descent"  
   https://arxiv.org/abs/2212.07677

2. **Garg et al. (2022)**: "What Can Transformers Learn In-Context?"  
   https://arxiv.org/abs/2208.01066

3. **Von Oswald et al. (2022)**: "Transformers Learn In-Context by Gradient Descent"  
   NeurIPS 2023
