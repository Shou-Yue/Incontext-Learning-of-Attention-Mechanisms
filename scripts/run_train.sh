#!/bin/bash
# Convenience script for running training

# Usage:
#   ./scripts/run_train.sh test_local    # Test locally
#   ./scripts/run_train.sh full          # Full training

set -e

CONFIG_NAME=${1:-test_local}

case $CONFIG_NAME in
  test_local)
    CONFIG_FILE="configs/test_local.json"
    ;;
  full)
    CONFIG_FILE="configs/train_linear_20d.json"
    ;;
  *)
    echo "Unknown config: $CONFIG_NAME"
    echo "Usage: $0 [test_local|full]"
    exit 1
    ;;
esac

echo "Running training with config: $CONFIG_FILE"
python src/training/train.py --config $CONFIG_FILE
