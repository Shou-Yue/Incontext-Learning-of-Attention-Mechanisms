#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIR="${1:-latest}"
LOWRANK_MODE="${2:-auto}"
FRESH_PULL="${3:-}"  # pass --fresh to wipe local cache before pulling

LOCAL_RESULTS_ROOT="$ROOT_DIR/results/from_dsmlp/results"

if [[ "$FRESH_PULL" == "--fresh" ]]; then
  echo "[0/2] Removing local cache: $LOCAL_RESULTS_ROOT"
  rm -rf "$LOCAL_RESULTS_ROOT"
fi

echo "[1/2] Pulling results from DSMLP..."
"$ROOT_DIR/pull-results.sh"

if [[ "$EXP_DIR" == "latest" ]]; then
  EXP_DIR="$(LOCAL_RESULTS_ROOT="$LOCAL_RESULTS_ROOT" python3 - <<'PY'
import glob, os
root = os.path.join(os.environ.get('LOCAL_RESULTS_ROOT', ''), '')
best = None
best_m = -1
for exp_dir in glob.glob(os.path.join(root, 'experiments_*')):
    if not os.path.isdir(exp_dir):
        continue
    latest = -1
    for cur_root, _, files in os.walk(exp_dir):
        for f in files:
            if f == 'all_results.json':
                p = os.path.join(cur_root, f)
                try:
                    m = os.path.getmtime(p)
                    if m > latest:
                        latest = m
                except FileNotFoundError:
                    pass
    if latest < 0:
        # fallback to dir mtime
        try:
            latest = os.path.getmtime(exp_dir)
        except FileNotFoundError:
            continue
    if latest > best_m:
        best_m = latest
        best = exp_dir
print(best or '')
PY
  )"
  if [[ -z "$EXP_DIR" ]]; then
    echo "No experiments_* folders found under $LOCAL_RESULTS_ROOT"
    exit 1
  fi
else
  if [[ -d "$EXP_DIR" ]]; then
    :
  else
    EXP_DIR="$LOCAL_RESULTS_ROOT/$EXP_DIR"
  fi
fi

if [[ ! -d "$EXP_DIR" ]]; then
  echo "Experiment folder not found: $EXP_DIR"
  exit 1
fi

echo "[2/2] Plotting interactive HTML from: $EXP_DIR"
python3 "$ROOT_DIR/scripts/plot_interactive_experiments.py" \
  --steps_results "$EXP_DIR/exp_steps_sweep/all_results.json" \
  --layers_results "$EXP_DIR/exp_layers_sweep/all_results.json" \
  --context_results "$EXP_DIR/exp_context_sweep/all_results.json" \
  --context_zero_results "$EXP_DIR/exp_context_sweep_zero/all_results.json" \
  --output_dir "$EXP_DIR/plots" \
  --lowrank_mode "$LOWRANK_MODE"

echo "Done. Open: $EXP_DIR/plots/interactive_plots.html"
