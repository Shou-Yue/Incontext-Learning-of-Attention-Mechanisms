#!/bin/bash
# Retrieve LSA experiment results from AWS
#
# Usage:
#   ./scripts/aws_lsa_retrieve.sh

set -e

# AWS Configuration (hardcoded)
KEY_FILE="$HOME/.ssh/ask-hammerspace.pem"
INSTANCE_IP="18.118.222.12"
INSTANCE_USER="ec2-user"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=================================================="
echo "Retrieving LSA Results from AWS"
echo "=================================================="
echo "Instance: $INSTANCE_USER@$INSTANCE_IP"
echo ""

# Create local results directory
LOCAL_RESULTS_DIR="./results_aws_${TIMESTAMP}"
mkdir -p "$LOCAL_RESULTS_DIR"

echo "Step 1: Downloading results..."

# Download results directory
scp -i "$KEY_FILE" -r \
    "$INSTANCE_USER@$INSTANCE_IP:~/lsa-icl/results/lsa_multilayer" \
    "$LOCAL_RESULTS_DIR/"

echo ""
echo "Step 2: Downloading checkpoints (optional)..."
# Try to download checkpoints (may be large)
scp -i "$KEY_FILE" -r \
    "$INSTANCE_USER@$INSTANCE_IP:~/lsa-icl/results/lsa_multilayer/*.pt" \
    "$LOCAL_RESULTS_DIR/lsa_multilayer/" 2>/dev/null || echo "No checkpoints found or skipped"

echo ""
echo "=================================================="
echo "Download Complete!"
echo "=================================================="
echo ""
echo "Results saved to: $LOCAL_RESULTS_DIR"
echo ""
echo "View results:"
echo "  cat $LOCAL_RESULTS_DIR/lsa_multilayer/results_table.csv"
echo ""
echo "View plots:"
echo "  open $LOCAL_RESULTS_DIR/lsa_multilayer/lsa_vs_gd_combined.png"
echo ""
echo "Generate new plots locally:"
echo "  python scripts/plot_lsa_multilayer.py \\"
echo "    --results_file $LOCAL_RESULTS_DIR/lsa_multilayer/all_results.json \\"
echo "    --output_dir $LOCAL_RESULTS_DIR/lsa_multilayer"
echo ""
echo "=================================================="
