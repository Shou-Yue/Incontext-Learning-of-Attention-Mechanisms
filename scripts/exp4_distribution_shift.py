#!/usr/bin/env python3
"""Experiment 4: Distribution shift robustness."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.experiments.runner import evaluate_condition, save_experiment_result, train_and_get_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4 distribution shift")
    parser.add_argument("--config", type=str, default="configs/train_baseline_d20.json")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--output", type=str, default="results/exp4_shift")
    args = parser.parse_args()

    base = load_config(args.config)
    base["train"]["total_steps"] = args.steps
    base["model"]["task_dim"] = 20
    base["model"]["max_context"] = 40
    base["data"]["x_distribution"] = "isotropic"
    base["data"]["orthant_mode"] = "none"

    mechanisms = {
        "baseline": {"attn_type": "baseline"},
        "linear": {"attn_type": "linear"},
        "lowrank_r16": {"attn_type": "lowrank", "attn_rank": 16},
    }

    shift_settings = {
        "in_distribution": {"x_distribution": "isotropic", "orthant_mode": "none"},
        "skewed_covariance": {"x_distribution": "skewed", "orthant_mode": "none", "skew_power": 2.0},
        "orthant_split": {"x_distribution": "isotropic", "orthant_mode": "split"},
    }

    results = {}

    for mech_name, mech_cfg in mechanisms.items():
        cfg = deepcopy(base)
        cfg["model"].update(mech_cfg)
        cfg["output_dir"] = str(Path(args.output) / mech_name / "train")

        print(f"\n[Exp4] Training {mech_name}")
        ckpt = train_and_get_checkpoint(cfg)

        mech_results = {}
        for shift_name, data_override in shift_settings.items():
            metrics = evaluate_condition(
                config=cfg,
                checkpoint_path=ckpt,
                n_context=40,
                active_dim=20,
                data_overrides=data_override,
            )
            mech_results[shift_name] = metrics
            print(f"  {shift_name}: mse={metrics['mse']:.4f}, cos={metrics.get('cosine_alignment', float('nan')):.4f}")

        results[mech_name] = {
            "checkpoint": str(ckpt),
            "metrics": mech_results,
        }

    save_experiment_result(args.output, "results.json", results)

    shifts = list(shift_settings.keys())
    x_idx = range(len(shifts))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for mech in mechanisms:
        ys = [results[mech]["metrics"][s]["mse"] for s in shifts]
        plt.plot(x_idx, ys, marker="o", label=mech)
    plt.xticks(list(x_idx), shifts, rotation=20)
    plt.ylabel("MSE")
    plt.title("Exp4: MSE under shift")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for mech in mechanisms:
        ys = [results[mech]["metrics"][s].get("cosine_alignment", float("nan")) for s in shifts]
        plt.plot(x_idx, ys, marker="o", label=mech)
    plt.xticks(list(x_idx), shifts, rotation=20)
    plt.ylabel("Cosine to GD-1")
    plt.title("Exp4: Alignment under shift")
    plt.grid(alpha=0.3)

    out_path = Path(args.output) / "exp4_shift.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
