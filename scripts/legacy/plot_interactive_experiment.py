#!/usr/bin/env python3
"""
Interactive plot with method toggles (CheckButtons).

Usage:
  python scripts/plot_interactive_experiment.py \
    --results_file results/exp_all/exp_steps_sweep/all_results.json \
    --metric mse

Auto-detects x-axis by key:
  - train_steps
  - num_layers
  - n_points
"""
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons


def _collect_methods(results, metric_prefix):
    methods = set()
    for r in results:
        for k in r.keys():
            if k.startswith(metric_prefix) and k.endswith("_mean"):
                methods.add(k.replace(metric_prefix, "").replace("_mean", ""))
    methods.discard("gd")
    return sorted(methods)


def _label(method):
    if method == "lsa":
        return "LSA"
    if method == "softmax":
        return "Softmax"
    if method == "kernel":
        return "Kernelized Linear"
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


def _detect_x(results):
    sample = results[0] if results else {}
    if "train_steps" in sample:
        return "train_steps", "Training Steps"
    if "num_layers" in sample:
        return "num_layers", "Number of Layers / GD Steps"
    if "n_points" in sample:
        return "n_points", "In-context Examples (n)"
    raise ValueError("Could not detect x-axis key (train_steps/num_layers/n_points).")


def main():
    parser = argparse.ArgumentParser(description="Interactive plot for one experiment")
    parser.add_argument('--results_file', type=str, required=True)
    parser.add_argument('--metric', type=str, choices=['mse', 'cosine'], default='mse')
    args = parser.parse_args()

    results = json.loads(Path(args.results_file).read_text())
    results = sorted(results, key=lambda r: r.get('train_steps', r.get('num_layers', r.get('n_points', 0))))

    x_key, x_label = _detect_x(results)
    x = np.array([r.get(x_key, 0) for r in results])

    if args.metric == "mse":
        metric_prefix = "mse_"
        title = "MSE"
        y_label = "Mean Squared Error"
    else:
        metric_prefix = "cosine_sim_"
        title = "Cosine Similarity vs GD"
        y_label = "Cosine Similarity"

    methods = _collect_methods(results, metric_prefix)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.32)

    artists = {}
    colors = {
        "lsa": "#4472C4",
        "softmax": "#5B9BD5",
        "kernel": "#70AD47",
    }
    markers = ['o', '^', 'D', 'v', 'P', 'X', '*']

    # Baseline GD for MSE
    if args.metric == "mse":
        mse_gd, mse_gd_std = _extract_series(results, "mse_gd")
        line, caplines, barlinecols = ax.errorbar(
            x, mse_gd, yerr=mse_gd_std, fmt='-s', label="T-step GD", color="#ED7D31"
        )
        artists["T-step GD"] = [line, *caplines, *barlinecols]

    for i, m in enumerate(methods):
        vals, stds = _extract_series(results, f"{metric_prefix}{m}")
        color = colors.get(m, None)
        line, caplines, barlinecols = ax.errorbar(
            x, vals, yerr=stds,
            fmt=f'-{markers[i % len(markers)]}',
            label=_label(m),
            color=color
        )
        artists[_label(m)] = [line, *caplines, *barlinecols]

    if args.metric == "cosine":
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Perfect Alignment")

    ax.set_title(f"{title} vs {x_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Checkboxes
    labels = list(artists.keys())
    visibility = [True] * len(labels)
    rax = plt.axes([0.02, 0.3, 0.25, 0.5])
    check = CheckButtons(rax, labels, visibility)

    def _toggle(label):
        for art in artists[label]:
            art.set_visible(not art.get_visible())
        plt.draw()

    check.on_clicked(_toggle)
    plt.show()


if __name__ == '__main__':
    main()
