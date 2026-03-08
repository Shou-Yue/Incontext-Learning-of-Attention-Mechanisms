#!/usr/bin/env python3
"""
Interactive plot for all three experiments (6 subplots).

Layout:
  Row 1: training steps (MSE, Cosine)
  Row 2: layers (MSE, Cosine)
  Row 3: in-context examples (MSE, Cosine)

Provides checkboxes to toggle attention variants across all subplots.
"""
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons


def _collect_methods(results_list):
    methods = set()
    for results in results_list:
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


def _prepare(results, x_key):
    results = sorted(results, key=lambda r: r.get(x_key, 0))
    x = np.array([r.get(x_key, 0) for r in results])
    return results, x


def _plot_series(ax, x, y, yerr, label, color, marker):
    mask = np.isfinite(y)
    if mask.sum() == 0:
        return None, None
    line = ax.plot(x[mask], y[mask], marker=marker, label=label, color=color)[0]
    fill = ax.fill_between(
        x[mask],
        (y - yerr)[mask],
        (y + yerr)[mask],
        color=color,
        alpha=0.15
    )
    return line, fill


def main():
    parser = argparse.ArgumentParser(description="Interactive plots for all experiments")
    parser.add_argument('--steps_results', type=str, required=True)
    parser.add_argument('--layers_results', type=str, required=True)
    parser.add_argument('--context_results', type=str, required=True)
    args = parser.parse_args()

    steps = json.loads(Path(args.steps_results).read_text())
    layers = json.loads(Path(args.layers_results).read_text())
    context = json.loads(Path(args.context_results).read_text())

    methods = _collect_methods([steps, layers, context])

    # Prepare x-axes
    steps, x_steps = _prepare(steps, "train_steps")
    layers, x_layers = _prepare(layers, "num_layers")
    context, x_context = _prepare(context, "n_points")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    plt.subplots_adjust(left=0.28, hspace=0.35, wspace=0.25)

    # Titles and labels
    axes[0, 0].set_title("Steps: MSE")
    axes[0, 1].set_title("Steps: Cosine vs GD")
    axes[1, 0].set_title("Layers: MSE")
    axes[1, 1].set_title("Layers: Cosine vs GD")
    axes[2, 0].set_title("Context: MSE")
    axes[2, 1].set_title("Context: Cosine vs GD")

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean Squared Error")
    for ax in axes[:, 1]:
        ax.set_ylabel("Cosine Similarity")

    axes[2, 0].set_xlabel("In-context Examples (n)")
    axes[2, 1].set_xlabel("In-context Examples (n)")
    axes[0, 0].set_xlabel("Training Steps")
    axes[0, 1].set_xlabel("Training Steps")
    axes[1, 0].set_xlabel("Number of Layers / GD Steps")
    axes[1, 1].set_xlabel("Number of Layers / GD Steps")

    # Baselines
    for ax in axes[:, 1]:
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Perfect Alignment")

    # GD baseline on MSE plots
    for ax, results, x in [
        (axes[0, 0], steps, x_steps),
        (axes[1, 0], layers, x_layers),
        (axes[2, 0], context, x_context),
    ]:
        y, yerr = _extract_series(results, "mse_gd")
        _plot_series(ax, x, y, yerr, "T-step GD", "#ED7D31", "s")

    colors = {
        "lsa": "#4472C4",
        "softmax": "#5B9BD5",
        "kernel": "#70AD47",
    }
    markers = ['o', '^', 'D', 'v', 'P', 'X', '*']

    # Store artists for toggling
    artists = {m: [] for m in methods}

    for i, m in enumerate(methods):
        color = colors.get(m, None)
        marker = markers[i % len(markers)]

        # Steps sweep
        y, yerr = _extract_series(steps, f"mse_{m}")
        line, fill = _plot_series(axes[0, 0], x_steps, y, yerr, _label(m), color, marker)
        if line:
            artists[m].extend([line, fill])
        y, yerr = _extract_series(steps, f"cosine_sim_{m}")
        line, fill = _plot_series(axes[0, 1], x_steps, y, yerr, _label(m), color, marker)
        if line:
            artists[m].extend([line, fill])

        # Layers sweep
        y, yerr = _extract_series(layers, f"mse_{m}")
        line, fill = _plot_series(axes[1, 0], x_layers, y, yerr, _label(m), color, marker)
        if line:
            artists[m].extend([line, fill])
        y, yerr = _extract_series(layers, f"cosine_sim_{m}")
        line, fill = _plot_series(axes[1, 1], x_layers, y, yerr, _label(m), color, marker)
        if line:
            artists[m].extend([line, fill])

        # Context sweep
        y, yerr = _extract_series(context, f"mse_{m}")
        line, fill = _plot_series(axes[2, 0], x_context, y, yerr, _label(m), color, marker)
        if line:
            artists[m].extend([line, fill])
        y, yerr = _extract_series(context, f"cosine_sim_{m}")
        line, fill = _plot_series(axes[2, 1], x_context, y, yerr, _label(m), color, marker)
        if line:
            artists[m].extend([line, fill])

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    # Single legend
    axes[0, 0].legend(loc='best')

    # Checkboxes for methods
    rax = plt.axes([0.02, 0.3, 0.22, 0.5])
    labels = [_label(m) for m in methods]
    visibility = [True] * len(labels)
    check = CheckButtons(rax, labels, visibility)

    label_to_method = {lbl: m for lbl, m in zip(labels, methods)}

    def _toggle(label):
        method = label_to_method[label]
        for art in artists[method]:
            if art is None:
                continue
            art.set_visible(not art.get_visible())
        plt.draw()

    check.on_clicked(_toggle)
    plt.show()


if __name__ == '__main__':
    main()
