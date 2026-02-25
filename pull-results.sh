#!/usr/bin/env bash
set -euo pipefail

LOCAL_PROJECT_DIR="/Users/krish.prasad/Desktop/code/Incontext-Learning-of-Attention-Mechanisms"
REMOTE_USER="krprasad"
REMOTE_HOST="dsmlp-login.ucsd.edu"
REMOTE_PROJECT_DIR="~/DSC180B"
LOCAL_DEST_DIR="${LOCAL_PROJECT_DIR}/results/from_dsmlp"

mkdir -p "$LOCAL_DEST_DIR"

# Pull all experiment artifacts under remote results/
scp -r \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/results/" \
  "$LOCAL_DEST_DIR"

echo "Pulled DSMLP results to: ${LOCAL_DEST_DIR}/"
