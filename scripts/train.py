#!/usr/bin/env python3
"""Train one model configuration."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q2 attention model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.steps is not None:
        config["train"]["total_steps"] = args.steps
    if args.output is not None:
        config["output_dir"] = args.output

    trainer = Trainer(config)
    best = trainer.train()
    print("Training complete")
    if best:
        print(best)


if __name__ == "__main__":
    main()
