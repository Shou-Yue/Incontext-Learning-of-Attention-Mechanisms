# AWS Workflow for LSA Multi-layer Experiments

This guide walks you through running the deep LSA experiments (1, 2, 4, 8, 16 layers) on AWS GPU.

## Prerequisites

1. AWS EC2 instance with GPU (e.g., g6.2xlarge with NVIDIA L4)
2. Your AWS key pair (.pem file)
3. Instance IP address

## Complete Workflow

### Step 1: Deploy to AWS

```bash
./scripts/aws_lsa_deploy.sh ~/path/to/your-key.pem 54.123.45.67
```

This will:
- Package your local code
- Upload to AWS
- Extract and setup environment
- Install PyTorch with CUDA

### Step 2: SSH and Run Experiment

```bash
# SSH into instance
ssh -i ~/path/to/your-key.pem ubuntu@54.123.45.67

# Start tmux session (so training continues if you disconnect)
tmux new -s lsa

# Navigate to project and activate environment
cd ~/lsa-icl
source venv/bin/activate

# Run the experiment
python scripts/lsa_gd_multilayer.py \
  --d 20 \
  --num_layers_list 1 2 4 8 16 \
  --sigma 0.0 \
  --num_epochs 30 \
  --device cuda

# Detach from tmux: Press Ctrl+B, then D
```

### Step 3: Monitor Progress (Optional)

From your local machine or another terminal:

```bash
# SSH into instance
ssh -i ~/path/to/your-key.pem ubuntu@54.123.45.67

# Monitor GPU usage
watch -n 1 nvidia-smi

# Or reattach to tmux session
tmux attach -t lsa
```

### Step 4: Retrieve Results

Once training completes (you'll see the summary table):

```bash
# From your local machine
./scripts/aws_lsa_retrieve.sh ~/path/to/your-key.pem 54.123.45.67
```

This will download:
- All results to `results_aws_<timestamp>/`
- Plots (.png files)
- Metrics (all_results.json, results_table.csv)
- Model checkpoints (.pt files)

### Step 5: View Results Locally

```bash
# View summary
cat results_aws_*/lsa_multilayer/results_table.csv

# Open plots
open results_aws_*/lsa_multilayer/lsa_vs_gd_combined.png

# Or regenerate plots
python scripts/plot_lsa_multilayer.py \
  --results_file results_aws_*/lsa_multilayer/all_results.json \
  --output_dir results_aws_*/lsa_multilayer
```

## Expected Timeline

With g6.2xlarge (NVIDIA L4):
- **1-layer**: ~5 minutes
- **2-layer**: ~10 minutes  
- **4-layer**: ~20 minutes
- **8-layer**: ~40 minutes
- **16-layer**: ~80 minutes

**Total**: ~2.5 hours for all layers

## Troubleshooting

### If training crashes

```bash
# SSH back in
ssh -i ~/path/to/your-key.pem ubuntu@54.123.45.67

# Reattach to tmux
tmux attach -t lsa

# Check what happened in the logs
```

### If GPU not detected

```bash
# Check CUDA availability
cd ~/lsa-icl
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

### If out of memory

Reduce batch size:
```bash
python scripts/lsa_gd_multilayer.py \
  --d 20 \
  --num_layers_list 1 2 4 8 16 \
  --sigma 0.0 \
  --num_epochs 30 \
  --batch_size 32 \
  --device cuda
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./scripts/aws_lsa_deploy.sh <key> <ip>` | Deploy code to AWS |
| `ssh -i <key> ubuntu@<ip>` | Connect to AWS |
| `tmux new -s lsa` | Start persistent session |
| `Ctrl+B, then D` | Detach from tmux |
| `tmux attach -t lsa` | Reattach to session |
| `watch -n 1 nvidia-smi` | Monitor GPU |
| `./scripts/aws_lsa_retrieve.sh <key> <ip>` | Download results |

## Alternative: Run Locally (CPU)

If you don't want to use AWS, you can run on CPU (slower):

```bash
python scripts/lsa_gd_multilayer.py \
  --d 20 \
  --num_layers_list 1 2 4 8 16 \
  --sigma 0.0 \
  --num_epochs 30 \
  --device cpu
```

Expect ~6-8 hours runtime on CPU.
