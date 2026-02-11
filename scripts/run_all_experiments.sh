#!/usr/bin/env bash
set -euo pipefail

python scripts/exp1_baseline_comparison.py "$@"
python scripts/exp2_scaling.py "$@"
python scripts/exp3_noise_robustness.py "$@"
python scripts/exp4_distribution_shift.py "$@"
python scripts/exp5_lowrank_sweep.py "$@"
python scripts/exp6_attention_analysis.py "$@"
