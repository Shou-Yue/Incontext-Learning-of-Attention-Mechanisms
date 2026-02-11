#!/usr/bin/env python3
"""Experiment 5: Low-rank sweep."""
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
from src.data.task_sampler import LinearRegressionTaskSampler
from src.experiments.runner import evaluate_condition, load_model_from_checkpoint, save_experiment_result, train_and_get_checkpoint


def estimate_attention_rank(model, sampler, n_context, active_dim):
    model.eval()
    batch = sampler.sample_batch(
        batch_size=1,
        n_context=n_context,
        active_dim=active_dim,
        noise_std=0.0,
        x_distribution="isotropic",
        skew_power=2.0,
        orthant_mode="none",
    )
    with torch.no_grad():
        _, attn_list = model.forward(
            batch["x_context"],
            batch["y_context"],
            batch["x_query"],
            return_attn=True,
        )
    if not attn_list:
        return float("nan")
    # last layer, first head
    a = attn_list[-1][0, 0]  # (seq, seq)
    s = torch.linalg.svdvals(a)
    s = s / (s.max() + 1e-8)
    return int((s > 1e-3).sum().item())


def estimate_inference_ms(model, sampler, n_context, active_dim):
    model.eval()
    batch = sampler.sample_batch(
        batch_size=32,
        n_context=n_context,
        active_dim=active_dim,
        noise_std=0.0,
        x_distribution="isotropic",
        skew_power=2.0,
        orthant_mode="none",
    )

    with torch.no_grad():
        for _ in range(5):
            _ = model.predict(batch["x_context"], batch["y_context"], batch["x_query"])
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = model.predict(batch["x_context"], batch["y_context"], batch["x_query"])
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

    return 1000.0 * (end - start) / 20.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp5 lowrank rank sweep")
    parser.add_argument("--config", type=str, default="configs/train_lowrank_d20_r16.json")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--output", type=str, default="results/exp5_lowrank_sweep")
    args = parser.parse_args()

    base = load_config(args.config)
    base["train"]["total_steps"] = args.steps
    base["model"]["task_dim"] = 20
    base["model"]["max_context"] = 40
    base["model"]["attn_type"] = "lowrank"

    results = {}

    for rank in args.ranks:
        cfg = deepcopy(base)
        cfg["model"]["attn_rank"] = rank
        cfg["output_dir"] = str(Path(args.output) / f"rank_{rank}")

        print(f"\n[Exp5] Training low-rank r={rank}")
        ckpt = train_and_get_checkpoint(cfg)
        metrics = evaluate_condition(
            config=cfg,
            checkpoint_path=ckpt,
            n_context=40,
            active_dim=20,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_from_checkpoint(cfg, ckpt).to(device)
        sampler = LinearRegressionTaskSampler(task_dim=20, device=device)
        eff_rank = estimate_attention_rank(model, sampler, n_context=40, active_dim=20)
        inf_ms = estimate_inference_ms(model, sampler, n_context=40, active_dim=20)

        results[str(rank)] = {
            "metrics": metrics,
            "effective_attention_rank": eff_rank,
            "inference_ms": inf_ms,
            "checkpoint": str(ckpt),
        }
        print(f"  mse={metrics['mse']:.4f}, cos={metrics.get('cosine_alignment', float('nan')):.4f}, rank_eff={eff_rank}, inf_ms={inf_ms:.3f}")

    save_experiment_result(args.output, "results.json", results)

    ranks = args.ranks
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(ranks, [results[str(r)]["metrics"]["mse"] for r in ranks], marker="o")
    plt.xlabel("rank r")
    plt.ylabel("MSE")
    plt.title("Exp5: MSE vs rank")
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(ranks, [results[str(r)]["metrics"].get("cosine_alignment", float("nan")) for r in ranks], marker="o")
    plt.xlabel("rank r")
    plt.ylabel("Cosine to GD-1")
    plt.title("Exp5: Alignment vs rank")
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(ranks, [results[str(r)]["inference_ms"] for r in ranks], marker="o")
    plt.xlabel("rank r")
    plt.ylabel("Inference time (ms)")
    plt.title("Exp5: Efficiency vs rank")
    plt.grid(alpha=0.3)

    out_path = Path(args.output) / "exp5_rank_sweep.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
