#!/usr/bin/env bash
set -euo pipefail

LOCAL_PROJECT_DIR="/Users/krish.prasad/Desktop/Replicating-What-Can-Transformers-Learn-In-Context-Garg-et-al.-NeurIPS-2022-"
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
