#!/usr/bin/env python3
"""
Plot all four experiments:
  - training steps sweep
  - layer sweep
  - in-context sweep (trained)
  - in-context sweep (zero training)

Produces 8 graphs: MSE + cosine for each experiment.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _collect_methods(results):
    methods = set()
    for r in results:
        for k in r.keys():
            if k.startswith("mse_") and k.endswith("_mean") and k not in ("mse_gd_mean",):
                methods.add(k.replace("mse_", "").replace("_mean", ""))
    return sorted(methods)


def _label(method):
    if method == "lsa":
        return "LSA"
    if method == "softmax":
        return "Softmax"
    if method == "kernel":
        return "Kernelized Linear"
    if method == "gla":
        return "GLA"
    if method == "gqa":
        return "GQA"
    if method == "sparse":
        return "Sparse Causal"
    if method.startswith("lowrank_"):
        return method.replace("lowrank_", "Low-rank ")
    return method


def _extract_series(results, key_prefix):
    vals = []
    stds = []
    for r in results:
        vals.append(r.get(f"{key_prefix}_mean", np.nan))
        stds.append(r.get(f"{key_prefix}_std", np.nan))
    return np.array(vals, dtype=float), np.array(stds, dtype=float)


def _plot_experiment(results, x_key, x_label, out_prefix):
    results = sorted(results, key=lambda r: r.get(x_key, 0))
    x = np.array([r.get(x_key, 0) for r in results])

    methods = _collect_methods(results)
    colors = {
        "lsa": "#4472C4",
        "softmax": "#5B9BD5",
        "kernel": "#70AD47",
        "gla": "#9E480E",
        "gqa": "#8064A2",
        "sparse": "#2E75B6",
    }
    markers = ['o', '^', 'D', 'v', 'P', 'X', '*']

    # MSE plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    mse_gd, mse_gd_std = _extract_series(results, "mse_gd")
    ax1.errorbar(x, mse_gd, yerr=mse_gd_std, fmt='-s', label="T-step GD", color="#ED7D31")

    for i, m in enumerate(methods):
        mse_vals, mse_std = _extract_series(results, f"mse_{m}")
        color = colors.get(m, None)
        ax1.errorbar(
            x, mse_vals, yerr=mse_std,
            fmt=f'-{markers[i % len(markers)]}',
            label=_label(m),
            color=color
        )

    ax1.set_title(f"MSE vs {x_label}")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Mean Squared Error")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_mse.png", dpi=200)

    # Cosine plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Perfect Alignment")

    for i, m in enumerate(methods):
        cos_vals, cos_std = _extract_series(results, f"cosine_sim_{m}")
        color = colors.get(m, None)
        ax2.errorbar(
            x, cos_vals, yerr=cos_std,
            fmt=f'-{markers[i % len(markers)]}',
            label=f"{_label(m)} vs GD",
            color=color
        )

    ax2.set_title(f"Cosine Similarity vs {x_label}")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Cosine Similarity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_cosine.png", dpi=200)


def _load_if_exists(path_str):
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.exists():
        print(f"[warn] Results file not found: {p}")
        return None
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(description="Plot experiments (only those with available results)")
    parser.add_argument('--steps_results', type=str,
                        default='results/exp_all/exp_steps_sweep/all_results.json',
                        help='Path to steps sweep all_results.json')
    parser.add_argument('--layers_results', type=str,
                        default='results/exp_all/exp_layers_sweep/all_results.json',
                        help='Path to layers sweep all_results.json')
    parser.add_argument('--context_results', type=str,
                        default='results/exp_all/exp_context_sweep/all_results.json',
                        help='Path to context sweep all_results.json')
    parser.add_argument('--context_zero_results', type=str,
                        default='results/exp_all/exp_context_sweep_zero/all_results.json',
                        help='Path to zero-train context sweep all_results.json')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_results = _load_if_exists(args.steps_results)
    layers_results = _load_if_exists(args.layers_results)
    context_results = _load_if_exists(args.context_results)
    context_zero_results = _load_if_exists(args.context_zero_results)

    if steps_results is not None:
        _plot_experiment(
            steps_results,
            x_key="train_steps",
            x_label="Training Steps",
            out_prefix=str(output_dir / "steps")
        )
    if layers_results is not None:
        _plot_experiment(
            layers_results,
            x_key="num_layers",
            x_label="Number of Layers / GD Steps",
            out_prefix=str(output_dir / "layers")
        )
    if context_results is not None:
        _plot_experiment(
            context_results,
            x_key="n_points",
            x_label="In-context Examples (n)",
            out_prefix=str(output_dir / "context")
        )
    if context_zero_results is not None:
        _plot_experiment(
            context_zero_results,
            x_key="n_points",
            x_label="In-context Examples (n) (0 training)",
            out_prefix=str(output_dir / "context_zero")
        )

    print(f"Saved plots to: {output_dir}")


if __name__ == '__main__':
    main()
