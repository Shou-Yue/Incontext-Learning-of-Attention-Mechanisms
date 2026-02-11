#!/usr/bin/env python3
"""Quick smoke test across attention mechanisms."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.experiments.runner import evaluate_condition, train_and_get_checkpoint


def run_one(base_config, name, model_override):
    cfg = deepcopy(base_config)
    cfg["model"].update(model_override)
    cfg["train"]["total_steps"] = 200
    cfg["train"]["eval_interval"] = 100
    cfg["eval"]["num_prompts"] = 64
    cfg["eval"]["batch_size"] = 32
    cfg["output_dir"] = f"results/smoke/{name}"

    print(f"[smoke] {name}")
    ckpt = train_and_get_checkpoint(cfg)
    metrics = evaluate_condition(
        config=cfg,
        checkpoint_path=ckpt,
        n_context=10,
        active_dim=10,
    )
    print(metrics)


def main() -> None:
    base = load_config("configs/train_baseline_d20.json")

    run_one(base, "baseline", {"attn_type": "baseline"})
    run_one(base, "linear", {"attn_type": "linear"})
    run_one(base, "lowrank", {"attn_type": "lowrank", "attn_rank": 16})


if __name__ == "__main__":
    main()
