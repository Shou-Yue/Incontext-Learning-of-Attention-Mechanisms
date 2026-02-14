#!/usr/bin/env python3
"""
Plot results from lowrank_softmax_sweep.py.

Produces:
  - lowrank_sweep_mse.png
  - lowrank_sweep_cosine.png
  - lowrank_sweep_combined.png
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _extract_series(results, key_prefix):
    vals = []
    stds = []
    for r in results:
        vals.append(r.get(f"{key_prefix}_mean", np.nan))
        stds.append(r.get(f"{key_prefix}_std", np.nan))
    return np.array(vals), np.array(stds)


def main():
    parser = argparse.ArgumentParser(description="Plot low-rank vs softmax sweep")
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to all_results.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save plots')
    args = parser.parse_args()

    results_path = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'r') as f:
        results = json.load(f)

    results = sorted(results, key=lambda r: r.get('num_layers', 0))
    layers = np.array([r.get('num_layers', 0) for r in results])

    # Collect series keys
    series_keys = []
    if results:
        for k in results[0].keys():
            if k.startswith("mse_lowrank_k") and k.endswith("_mean"):
                series_keys.append(k.replace("_mean", "").replace("mse_", ""))
    series_keys = sorted(series_keys)

    # Build plots
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title("MSE Comparison")
    ax1.set_xlabel("Number of Layers / GD Steps")
    ax1.set_ylabel("Mean Squared Error")

    # GD
    mse_gd, mse_gd_std = _extract_series(results, "mse_gd")
    ax1.errorbar(layers, mse_gd, yerr=mse_gd_std, fmt='-s', label="T-step GD")

    # Softmax
    mse_soft, mse_soft_std = _extract_series(results, "mse_softmax")
    ax1.errorbar(layers, mse_soft, yerr=mse_soft_std, fmt='-o', label="Softmax")

    # Low-rank variants
    markers = ['^', 'v', 'D', 'P', 'X']
    for i, sk in enumerate(series_keys):
        mse_vals, mse_std = _extract_series(results, f"mse_{sk}")
        ax1.errorbar(layers, mse_vals, yerr=mse_std,
                     fmt=f'-{markers[i % len(markers)]}', label=sk.replace("lowrank_", "Low-rank "))

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title("Weight Update Alignment vs GD")
    ax2.set_xlabel("Number of Layers / GD Steps")
    ax2.set_ylabel("Cosine Similarity")
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Perfect Alignment")

    cos_soft, cos_soft_std = _extract_series(results, "cosine_sim_softmax")
    ax2.errorbar(layers, cos_soft, yerr=cos_soft_std, fmt='-o', label="Softmax vs GD")

    for i, sk in enumerate(series_keys):
        cos_vals, cos_std = _extract_series(results, f"cosine_sim_{sk}")
        ax2.errorbar(layers, cos_vals, yerr=cos_std,
                     fmt=f'-{markers[i % len(markers)]}', label=f"{sk.replace('lowrank_', 'Low-rank ')} vs GD")

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Combined plot
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))

    ax3a.set_title("MSE Comparison")
    ax3a.set_xlabel("Number of Layers / GD Steps")
    ax3a.set_ylabel("Mean Squared Error")
    ax3a.errorbar(layers, mse_gd, yerr=mse_gd_std, fmt='-s', label="T-step GD")
    ax3a.errorbar(layers, mse_soft, yerr=mse_soft_std, fmt='-o', label="Softmax")
    for i, sk in enumerate(series_keys):
        mse_vals, mse_std = _extract_series(results, f"mse_{sk}")
        ax3a.errorbar(layers, mse_vals, yerr=mse_std,
                      fmt=f'-{markers[i % len(markers)]}', label=sk.replace("lowrank_", "Low-rank "))
    ax3a.legend()
    ax3a.grid(True, alpha=0.3)

    ax3b.set_title("Weight Update Alignment vs GD")
    ax3b.set_xlabel("Number of Layers / GD Steps")
    ax3b.set_ylabel("Cosine Similarity")
    ax3b.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Perfect Alignment")
    ax3b.errorbar(layers, cos_soft, yerr=cos_soft_std, fmt='-o', label="Softmax vs GD")
    for i, sk in enumerate(series_keys):
        cos_vals, cos_std = _extract_series(results, f"cosine_sim_{sk}")
        ax3b.errorbar(layers, cos_vals, yerr=cos_std,
                      fmt=f'-{markers[i % len(markers)]}', label=f"{sk.replace('lowrank_', 'Low-rank ')} vs GD")
    ax3b.legend()
    ax3b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig1.savefig(output_dir / "lowrank_sweep_mse.png", dpi=200)
    fig2.savefig(output_dir / "lowrank_sweep_cosine.png", dpi=200)
    fig3.savefig(output_dir / "lowrank_sweep_combined.png", dpi=200)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
