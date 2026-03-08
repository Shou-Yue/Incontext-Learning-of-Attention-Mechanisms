#!/bin/bash
# Deploy LSA experiment to AWS
#
# Usage:
#   ./scripts/aws_lsa_deploy.sh

set -e

# AWS Configuration (hardcoded)
KEY_FILE="$HOME/.ssh/ask-hammerspace.pem"
INSTANCE_IP="18.118.222.12"
INSTANCE_USER="ec2-user"
PROJECT_NAME="lsa-icl"

echo "=================================================="
echo "AWS LSA Experiment Deployment"
echo "=================================================="
echo "Key: $KEY_FILE"
echo "Instance: $INSTANCE_USER@$INSTANCE_IP"
echo ""

# Step 1: Package the project
echo "Step 1: Packaging project..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="${PROJECT_NAME}_${TIMESTAMP}.tar.gz"

tar -czf "/tmp/$ARCHIVE_NAME" \
    --exclude=".git" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="checkpoints" \
    --exclude="results" \
    --exclude="*.png" \
    --exclude="*.pt" \
    .

echo "Created archive: /tmp/$ARCHIVE_NAME"

# Step 2: Upload to AWS
echo ""
echo "Step 2: Uploading to AWS..."
scp -i "$KEY_FILE" "/tmp/$ARCHIVE_NAME" "$INSTANCE_USER@$INSTANCE_IP:~/"
echo "Upload complete!"

# Step 3: Setup on AWS
echo ""
echo "Step 3: Setting up environment on AWS..."
ssh -i "$KEY_FILE" "$INSTANCE_USER@$INSTANCE_IP" << 'ENDSSH'
    # Extract archive
    ARCHIVE=$(ls -t lsa-icl_*.tar.gz | head -1)
    echo "Extracting $ARCHIVE..."
    
    # Create project directory if it doesn't exist
    mkdir -p ~/lsa-icl
    cd ~/lsa-icl
    tar -xzf ~/"$ARCHIVE"
    
    # Run AWS setup script
    echo "Running AWS setup..."
    bash scripts/aws_setup.sh
    
    echo ""
    echo "Setup complete on AWS!"
ENDSSH

# Step 4: Provide run instructions
echo ""
echo "=================================================="
echo "Deployment Complete!"
echo "=================================================="
echo ""
echo "To run the experiment on AWS:"
echo ""
echo "1. SSH into instance:"
echo "   ssh -i $KEY_FILE $INSTANCE_USER@$INSTANCE_IP"
echo ""
echo "2. Start tmux session:"
echo "   tmux new -s lsa"
echo ""
echo "3. Navigate and activate environment:"
echo "   cd ~/lsa-icl"
echo "   source venv/bin/activate"
echo ""
echo "4. Run the experiment:"
echo "   python scripts/lsa_gd_multilayer.py \\"
echo "     --d 20 \\"
echo "     --num_layers_list 1 2 4 8 16 \\"
echo "     --sigma 0.0 \\"
echo "     --num_epochs 30 \\"
echo "     --device cuda"
echo ""
echo "5. Detach from tmux: Ctrl+B, then D"
echo ""
echo "6. Monitor progress (from another terminal):"
echo "   ssh -i $KEY_FILE $INSTANCE_USER@$INSTANCE_IP"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "7. When done, retrieve results:"
echo "   ./scripts/aws_lsa_retrieve.sh"
echo ""
echo "=================================================="
