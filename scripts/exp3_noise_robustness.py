#!/usr/bin/env python3
"""Experiment 3: Robustness to label noise."""
from __future__ import annotations

import argparse
import math
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
    parser = argparse.ArgumentParser(description="Exp3 noise robustness")
    parser.add_argument("--config", type=str, default="configs/train_baseline_d20.json")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--noise_var", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23, 31, 47])
    parser.add_argument("--output", type=str, default="results/exp3_noise")
    args = parser.parse_args()

    base = load_config(args.config)
    base["train"]["total_steps"] = args.steps
    base["model"]["task_dim"] = 20
    base["model"]["max_context"] = 40

    mechanisms = {
        "baseline": {"attn_type": "baseline"},
        "linear": {"attn_type": "linear"},
        "lowrank_r16": {"attn_type": "lowrank", "attn_rank": 16},
    }

    results = {m: {} for m in mechanisms}

    for var in args.noise_var:
        std = math.sqrt(var)
        for mech_name, mech_cfg in mechanisms.items():
            seed_metrics = []
            for seed in args.seeds:
                cfg = deepcopy(base)
                cfg["seed"] = seed
                cfg["model"].update(mech_cfg)
                cfg["data"]["noise_std"] = std
                cfg["output_dir"] = str(Path(args.output) / mech_name / f"noise_{var}_seed_{seed}")

                print(f"\n[Exp3] Training {mech_name}, var={var}, seed={seed}")
                ckpt = train_and_get_checkpoint(cfg)
                metrics = evaluate_condition(
                    config=cfg,
                    checkpoint_path=ckpt,
                    n_context=40,
                    active_dim=20,
                )
                seed_metrics.append(metrics)

            # Aggregate over seeds
            mse_mean = sum(m["mse"] for m in seed_metrics) / len(seed_metrics)
            cos_mean = sum(m.get("cosine_alignment", 0.0) for m in seed_metrics) / len(seed_metrics)
            results[mech_name][str(var)] = {
                "mse_mean": mse_mean,
                "cosine_mean": cos_mean,
                "seed_metrics": seed_metrics,
            }
            print(f"[Exp3] {mech_name}, var={var}: mse={mse_mean:.4f}, cos={cos_mean:.4f}")

    save_experiment_result(args.output, "results.json", results)

    vars_ = [str(v) for v in args.noise_var]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for mech in mechanisms:
        ys = [results[mech][v]["mse_mean"] for v in vars_]
        plt.plot(args.noise_var, ys, marker="o", label=mech)
    plt.xlabel("noise variance")
    plt.ylabel("MSE")
    plt.title("Exp3: MSE vs noise")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for mech in mechanisms:
        ys = [results[mech][v]["cosine_mean"] for v in vars_]
        plt.plot(args.noise_var, ys, marker="o", label=mech)
    plt.xlabel("noise variance")
    plt.ylabel("Cosine to GD-1")
    plt.title("Exp3: Alignment vs noise")
    plt.grid(alpha=0.3)

    out_path = Path(args.output) / "exp3_noise.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
