#!/bin/bash
# AWS Setup Script for g6.2xlarge instance

set -e

echo "=== AWS Instance Setup for In-Context Learning ==="

# Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip if not already installed
echo "Installing Python and dependencies..."
sudo apt-get install -y python3-pip python3-venv git tmux htop

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Verify CUDA
echo "Verifying CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run training in tmux (recommended):"
echo "  1. Start tmux: tmux new -s training"
echo "  2. Activate venv: source venv/bin/activate"
echo "  3. Run training: python src/training/train.py --config configs/train_linear_20d.json"
echo "  4. Detach from tmux: Ctrl+B, then D"
echo "  5. Reattach later: tmux attach -t training"
echo ""
echo "To monitor GPU usage: watch -n 1 nvidia-smi"
