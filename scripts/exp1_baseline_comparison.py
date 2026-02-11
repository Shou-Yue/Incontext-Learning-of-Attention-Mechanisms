#!/usr/bin/env python3
"""Experiment 1: Baseline comparison across attention mechanisms."""
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
    parser = argparse.ArgumentParser(description="Exp1 baseline comparison")
    parser.add_argument("--config", type=str, default="configs/train_baseline_d20.json")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--contexts", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--lowrank_ranks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--eval_prompts", type=int, default=None)
    parser.add_argument("--eval_batch", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/exp1_baseline_comparison")
    args = parser.parse_args()

    base = load_config(args.config)
    base["train"]["total_steps"] = args.steps
    if args.eval_prompts is not None:
        base["eval"]["num_prompts"] = args.eval_prompts
    if args.eval_batch is not None:
        base["eval"]["batch_size"] = args.eval_batch
    if args.d_model is not None:
        base["model"]["d_model"] = args.d_model
    if args.n_layers is not None:
        base["model"]["n_layers"] = args.n_layers
    if args.n_heads is not None:
        base["model"]["n_heads"] = args.n_heads

    mechanisms = [
        ("softmax", {"model": {"attn_type": "baseline"}}),
        ("linear", {"model": {"attn_type": "linear", "attn_feature_map": "elu"}}),
    ]
    for rank in args.lowrank_ranks:
        mechanisms.append((f"lowrank_r{rank}", {"model": {"attn_type": "lowrank", "attn_rank": rank}}))

    results = {}

    for name, override in mechanisms:
        cfg = deepcopy(base)
        cfg["output_dir"] = str(Path(args.output) / name)
        for key, value in override.items():
            cfg[key].update(value)

        print(f"\n[Exp1] Training {name}")
        ckpt = train_and_get_checkpoint(cfg)

        mech_results = {}
        for n_context in args.contexts:
            metrics = evaluate_condition(
                config=cfg,
                checkpoint_path=ckpt,
                n_context=n_context,
                active_dim=cfg["model"]["task_dim"],
            )
            mech_results[str(n_context)] = metrics
            print(f"  n={n_context}: mse={metrics['mse']:.4f}, cos={metrics.get('cosine_alignment', float('nan')):.4f}")

        results[name] = {
            "checkpoint": str(ckpt),
            "metrics": mech_results,
        }

    save_experiment_result(args.output, "results.json", results)

    # Plot MSE and cosine alignment.
    contexts = [str(n) for n in args.contexts]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    gd_curve = [results["softmax"]["metrics"][c]["gd_one_step_mse"] for c in contexts]
    plt.plot(args.contexts, gd_curve, marker="x", linestyle="--", color="black", label="gd_one_step")
    for name in results:
        ys = [results[name]["metrics"][c]["mse"] for c in contexts]
        plt.plot(args.contexts, ys, marker="o", label=name)
    plt.xlabel("n in-context examples")
    plt.ylabel("MSE")
    plt.title("Exp1: MSE vs n")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for name in results:
        ys = [results[name]["metrics"][c].get("cosine_alignment", float("nan")) for c in contexts]
        plt.plot(args.contexts, ys, marker="o", label=name)
    plt.xlabel("n in-context examples")
    plt.ylabel("Cosine to GD-1")
    plt.title("Exp1: GD Alignment vs n")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    out_path = Path(args.output) / "exp1_panels.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
