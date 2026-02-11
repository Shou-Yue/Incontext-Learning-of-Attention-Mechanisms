#!/usr/bin/env python3
"""Evaluate a trained checkpoint under a specific condition."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.experiments.runner import evaluate_condition


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Q2 attention model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_context", type=int, required=True)
    parser.add_argument("--active_dim", type=int, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--x_distribution", type=str, default=None)
    parser.add_argument("--orthant_mode", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = args.checkpoint
    if checkpoint is None:
        best = Path(config["output_dir"]) / "checkpoints" / "checkpoint_best.pt"
        checkpoint = str(best if best.exists() else Path(config["output_dir"]) / "checkpoints" / "checkpoint_latest.pt")

    data_overrides = {}
    if args.noise_std is not None:
        data_overrides["noise_std"] = args.noise_std
    if args.x_distribution is not None:
        data_overrides["x_distribution"] = args.x_distribution
    if args.orthant_mode is not None:
        data_overrides["orthant_mode"] = args.orthant_mode

    metrics = evaluate_condition(
        config=config,
        checkpoint_path=checkpoint,
        n_context=args.n_context,
        active_dim=args.active_dim or config["model"]["task_dim"],
        data_overrides=data_overrides,
    )
    print(metrics)


if __name__ == "__main__":
    main()
