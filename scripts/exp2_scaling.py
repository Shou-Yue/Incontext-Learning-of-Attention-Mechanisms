#!/usr/bin/env python3
"""Experiment 2: Scaling with task dimension."""
from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.experiments.runner import evaluate_condition, save_experiment_result, train_and_get_checkpoint


def estimate_inference_ms(config, checkpoint, n_context, active_dim):
    from src.experiments.runner import load_model_from_checkpoint
    from src.data.task_sampler import LinearRegressionTaskSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(config, checkpoint).to(device)
    model.eval()
    sampler = LinearRegressionTaskSampler(task_dim=config["model"]["task_dim"], device=device)
    batch = sampler.sample_batch(
        batch_size=32,
        n_context=n_context,
        active_dim=active_dim,
        noise_std=0.0,
        x_distribution="isotropic",
        skew_power=2.0,
        orthant_mode="none",
    )

    x_context = batch["x_context"]
    y_context = batch["y_context"]
    x_query = batch["x_query"]

    with torch.no_grad():
        for _ in range(5):
            _ = model.predict(x_context, y_context, x_query)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = model.predict(x_context, y_context, x_query)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    return 1000.0 * (end - start) / 20.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2 scaling analysis")
    parser.add_argument("--config", type=str, default="configs/train_baseline_d20.json")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--output", type=str, default="results/exp2_scaling")
    args = parser.parse_args()

    base = load_config(args.config)
    base["train"]["total_steps"] = args.steps

    mechanisms = [
        "baseline",
        "linear",
        "lowrank",
    ]

    results = {m: {} for m in mechanisms}

    for dim in args.dims:
        for mech in mechanisms:
            cfg = deepcopy(base)
            cfg["model"]["task_dim"] = dim
            cfg["model"]["max_context"] = 2 * dim
            cfg["train"]["curriculum"]["end_dim"] = dim
            cfg["train"]["curriculum"]["end_context"] = 2 * dim

            if mech == "baseline":
                cfg["model"]["attn_type"] = "baseline"
                label = f"baseline_d{dim}"
            elif mech == "linear":
                cfg["model"]["attn_type"] = "linear"
                label = f"linear_d{dim}"
            else:
                cfg["model"]["attn_type"] = "lowrank"
                cfg["model"]["attn_rank"] = max(2, min(cfg["model"]["d_model"] // cfg["model"]["n_heads"], dim // 4))
                label = f"lowrank_d{dim}"

            cfg["output_dir"] = str(Path(args.output) / label)

            print(f"\n[Exp2] Training {label}")
            ckpt = train_and_get_checkpoint(cfg)
            metrics = evaluate_condition(
                config=cfg,
                checkpoint_path=ckpt,
                n_context=2 * dim,
                active_dim=dim,
            )
            inf_ms = estimate_inference_ms(cfg, ckpt, n_context=2 * dim, active_dim=dim)
            metrics["inference_ms"] = inf_ms
            results[mech][str(dim)] = metrics
            print(f"  mse={metrics['mse']:.4f}, cos={metrics.get('cosine_alignment', float('nan')):.4f}, inf_ms={inf_ms:.3f}")

    save_experiment_result(args.output, "results.json", results)

    plt.figure(figsize=(14, 4))
    dims = args.dims

    plt.subplot(1, 3, 1)
    for mech in mechanisms:
        ys = [results[mech][str(d)]["mse"] for d in dims]
        plt.plot(dims, ys, marker="o", label=mech)
    plt.xlabel("task dimension d")
    plt.ylabel("MSE")
    plt.title("Exp2: MSE vs d")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    plt.subplot(1, 3, 2)
    for mech in mechanisms:
        ys = [results[mech][str(d)].get("cosine_alignment", float("nan")) for d in dims]
        plt.plot(dims, ys, marker="o", label=mech)
    plt.xlabel("task dimension d")
    plt.ylabel("Cosine to GD-1")
    plt.title("Exp2: Alignment vs d")
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 3)
    for mech in mechanisms:
        ys = [results[mech][str(d)]["inference_ms"] for d in dims]
        plt.plot(dims, ys, marker="o", label=mech)
    plt.xlabel("task dimension d")
    plt.ylabel("Inference time (ms)")
    plt.title("Exp2: Efficiency vs d")
    plt.grid(alpha=0.3)

    out_path = Path(args.output) / "exp2_scaling.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
