#!/usr/bin/env bash
set -euo pipefail

LOCAL_PROJECT_DIR="/Users/krish.prasad/Desktop/code/Incontext-Learning-of-Attention-Mechanisms"
REMOTE_USER="krprasad"
REMOTE_HOST="dsmlp-login.ucsd.edu"
REMOTE_DIR="~/DSC180B"

cd "$LOCAL_PROJECT_DIR"

scp -r \
  README.md \
  requirements.txt \
  configs \
  src \
  scripts \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
