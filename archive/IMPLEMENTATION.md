# Implementation Guide

## What We've Built

We've implemented a complete training pipeline for replicating the paper **"What Can Transformers Learn In-Context?"** focusing on linear functions (20D).

### Core Components

1. **Data Generation** (`src/data/`)
   - `samplers.py`: Gaussian sampler for input generation
   - `tasks.py`: Linear regression task implementation
   - `curriculum.py`: Curriculum learning scheduler

2. **Model** (`src/models/`)
   - `transformer.py`: GPT-2 style Transformer (12 layers, 8 heads, 256 dim, ~22.4M parameters)
   - Interleaves input-output pairs for in-context learning
   - Auto-regressive prediction at each position

3. **Training** (`src/training/`)
   - `train.py`: Complete training loop with checkpointing
   - Supports curriculum learning
   - Auto-resumes from checkpoints
   - Logs to JSON lines file

4. **Configurations** (`configs/`)
   - `test_local.json`: Small model, 1k steps (for local testing)
   - `train_linear_20d.json`: Full model, 500k steps (for AWS)

## Model Architecture Details

```
Input dimension: 20
Max sequence length: 41 points (up to 40 in-context examples + 1 query)
Embedding dimension: 256
Transformer layers: 12
Attention heads: 8
Total parameters: ~22.4M

Sequence format: [x1, y1, x2, y2, ..., xk, yk, x_query]
where yi is padded with zeros to match dimension of xi
```

## Curriculum Learning Schedule

Starts simple and gradually increases complexity:
- **Dimensions**: 5 → 20 (increment by 1 every 2000 steps)
- **Points**: 11 → 41 (increment by 2 every 2000 steps)
- Reaches full complexity at step 32,000

## Training Time Estimates

### AWS g6.2xlarge (1x NVIDIA L4, 24GB VRAM)

| Steps | Estimated Time | Purpose |
|-------|---------------|---------|
| 1,000 | 1-2 hours | Quick validation |
| 10,000 | 12-24 hours | Initial assessment |
| 50,000 | 3-6 days | Performance check |
| 500,000 | **3-6 days** | Full replication |

*Note: Actual time depends on implementation efficiency and curriculum stage*

## Next Steps

### Phase 1: Local Testing (Today)
1. Install dependencies locally
2. Run small test (1k steps)
3. Verify everything works

### Phase 2: AWS Validation (1-2 days)
1. Transfer code to AWS
2. Run 10k-50k step training
3. Verify GPU utilization and training speed
4. Estimate actual time for full training

### Phase 3: Full Training (3-6 days)
1. Launch 500k step training
2. Monitor progress periodically
3. Let it run to completion

### Phase 4: Evaluation
1. Implement baselines (least squares, k-NN, averaging)
2. Evaluate on test prompts
3. Generate plots (error vs in-context examples)
4. Test distribution shifts

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run local test (1k steps, small model)
python src/training/train.py --config configs/test_local.json

# Or use convenience script
./scripts/run_train.sh test_local
```

Expected output:
- Model initialization with parameter count
- Training progress bar with loss
- Checkpoints saved every 500 steps
- Log file: `checkpoints/test_local/training_log.jsonl`

## AWS Deployment

### 1. Transfer Code
```bash
# On local machine
cd /Users/krish.prasad/Desktop/replication-in-context
tar -czf replication.tar.gz .

# Upload to AWS (replace with your instance IP)
scp -i your-key.pem replication.tar.gz ubuntu@your-instance:~
```

### 2. Setup AWS Instance
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance

# Extract and setup
tar -xzf replication.tar.gz
cd replication-in-context
bash scripts/aws_setup.sh
```

### 3. Run Training
```bash
# Start tmux session (so training continues if you disconnect)
tmux new -s training

# Activate environment
source venv/bin/activate

# Run full training
python src/training/train.py --config configs/train_linear_20d.json

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### 4. Monitor Progress
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Tail log file
tail -f checkpoints/linear_20d_full/training_log.jsonl

# Check latest loss
tail -1 checkpoints/linear_20d_full/training_log.jsonl | python -m json.tool
```

## Checkpoint Management

Checkpoints are saved:
- **Always**: `checkpoint_latest.pt` (overwritten each save)
- **Periodic**: `checkpoint_10000.pt`, `checkpoint_50000.pt`, etc. (kept permanently every 50k steps)

To resume from a checkpoint, just run training again with the same config - it auto-resumes.

## What's Not Implemented Yet

1. **Evaluation Module** (`src/evaluation/`)
   - Baseline implementations (least squares, k-NN, averaging)
   - Test set generation
   - Metrics computation
   - Plotting functions

2. **Distribution Shifts**
   - Skewed covariance testing
   - Noisy labels testing
   - Different orthants testing

3. **Advanced Analysis**
   - Function visualization
   - Gradient analysis
   - Robustness experiments

We can implement these once the model is trained.

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `n_embd`, `n_layer`, or `n_head`

### Slow Training
- Verify GPU is being used: check if "Using device: cuda" appears
- Monitor GPU utilization: `nvidia-smi`
- Ensure CUDA PyTorch is installed

### Loss Not Decreasing
- Check curriculum is working (dims and points should increase over time)
- Verify data generation is correct
- Try reducing learning rate

## Questions to Answer After Initial Training

1. **Does loss decrease smoothly?** Should drop as curriculum increases complexity
2. **What's the final loss value?** Paper shows very low error (~0.02 at 20 examples)
3. **How long per step?** Helps estimate full training time
4. **GPU utilization?** Should be >80% during training

## References

- Paper: https://arxiv.org/abs/2208.01066
- Original code: https://github.com/dtsip/in-context-learning
- Our implementation: Hybrid approach (cleaner, more modular)
