#!/usr/bin/env python3
"""
Plotting script for multi-layer LSA experiments.

Creates visualizations comparing LSA and GD performance across different depths.
"""
import os
import sys
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


def _collect_lowrank_keys(results, prefix):
    keys = set()
    target = f"{prefix}lowrank_"
    for r in results:
        for k in r.keys():
            if k.startswith(target) and k.endswith("_mean"):
                keys.add(k.replace("_mean", "").replace(prefix, ""))
    return sorted(keys)

def plot_mse_comparison(results, output_path):
    """
    Plot MSE comparison between LSA and GD across different num_layers.
    
    Args:
        results: list of result dicts from experiments
        output_path: path to save the plot
    """
    num_layers_list = [r['num_layers'] for r in results]
    
    lsa_means = np.array([r.get('mse_lsa_mean', np.nan) for r in results], dtype=float)
    lsa_stds = np.array([r.get('mse_lsa_std', np.nan) for r in results], dtype=float)
    gd_means = np.array([r.get('mse_gd_mean', np.nan) for r in results], dtype=float)
    gd_stds = np.array([r.get('mse_gd_std', np.nan) for r in results], dtype=float)

    softmax_means = np.array([r.get('mse_softmax_mean', np.nan) for r in results], dtype=float)
    softmax_stds = np.array([r.get('mse_softmax_std', np.nan) for r in results], dtype=float)
    kernel_means = np.array([r.get('mse_kernel_mean', np.nan) for r in results], dtype=float)
    kernel_stds = np.array([r.get('mse_kernel_std', np.nan) for r in results], dtype=float)

    lowrank_keys = _collect_lowrank_keys(results, "mse_")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot LSA
    if np.isfinite(lsa_means).any():
        ax.errorbar(num_layers_list, lsa_means, yerr=lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA', color='#4472C4')

    # Plot GD
    if np.isfinite(gd_means).any():
        ax.errorbar(num_layers_list, gd_means, yerr=gd_stds,
                    marker='s', linewidth=2, markersize=8, capsize=5,
                    label='T-step GD', color='#ED7D31')

    # Plot Softmax
    if np.isfinite(softmax_means).any():
        ax.errorbar(num_layers_list, softmax_means, yerr=softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax', color='#5B9BD5')

    # Plot Low-rank variants
    markers = ['v', 'P', 'X', '*']
    for i, sk in enumerate(lowrank_keys):
        mse_vals, mse_std = _extract_series(results, f"mse_{sk}")
        ax.errorbar(num_layers_list, mse_vals, yerr=mse_std,
                    marker=markers[i % len(markers)], linewidth=2, markersize=8, capsize=5,
                    label=sk.replace("lowrank_", "Low-rank "), color='#C00000')

    # Plot Kernelized Linear
    if np.isfinite(kernel_means).any():
        ax.errorbar(num_layers_list, kernel_means, yerr=kernel_stds,
                    marker='D', linewidth=2, markersize=8, capsize=5,
                    label='Kernelized Linear', color='#70AD47')
    
    ax.set_xlabel('Number of Layers / GD Steps', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    ax.set_title('LSA vs Multi-step Gradient Descent: MSE Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(num_layers_list)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved MSE comparison plot to {output_path}")
    plt.close()


def plot_cosine_similarity(results, output_path):
    """
    Plot cosine similarity between LSA and GD weight updates across different num_layers.
    
    Args:
        results: list of result dicts from experiments
        output_path: path to save the plot
    """
    num_layers_list = [r['num_layers'] for r in results]
    
    cos_lsa_means = np.array([r.get('cosine_sim_lsa_mean', r.get('cosine_sim_mean', np.nan)) for r in results], dtype=float)
    cos_lsa_stds = np.array([r.get('cosine_sim_lsa_std', r.get('cosine_sim_std', np.nan)) for r in results], dtype=float)
    cos_softmax_means = np.array([r.get('cosine_sim_softmax_mean', np.nan) for r in results], dtype=float)
    cos_softmax_stds = np.array([r.get('cosine_sim_softmax_std', np.nan) for r in results], dtype=float)
    cos_kernel_means = np.array([r.get('cosine_sim_kernel_mean', np.nan) for r in results], dtype=float)
    cos_kernel_stds = np.array([r.get('cosine_sim_kernel_std', np.nan) for r in results], dtype=float)

    lowrank_keys = _collect_lowrank_keys(results, "cosine_sim_")
    markers = ['v', 'P', 'X', '*']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cosine similarity (LSA baseline)
    if np.isfinite(cos_lsa_means).any():
        ax.errorbar(num_layers_list, cos_lsa_means, yerr=cos_lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA vs GD', color='#4472C4')

    if np.isfinite(cos_softmax_means).any():
        ax.errorbar(num_layers_list, cos_softmax_means, yerr=cos_softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax vs GD', color='#5B9BD5')

    for i, sk in enumerate(lowrank_keys):
        cos_vals, cos_std = _extract_series(results, f"cosine_sim_{sk}")
        ax.errorbar(num_layers_list, cos_vals, yerr=cos_std,
                    marker=markers[i % len(markers)], linewidth=2, markersize=8, capsize=5,
                    label=f"{sk.replace('lowrank_', 'Low-rank ')} vs GD", color='#C00000')

    if np.isfinite(cos_kernel_means).any():
        ax.errorbar(num_layers_list, cos_kernel_means, yerr=cos_kernel_stds,
                    marker='D', linewidth=2, markersize=8, capsize=5,
                    label='Kernelized Linear vs GD', color='#70AD47')
    
    # Add reference line at 1.0 (perfect alignment)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Alignment')
    
    ax.set_xlabel('Number of Layers / GD Steps', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('Weight Update Alignment vs GD', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(num_layers_list)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved cosine similarity plot to {output_path}")
    plt.close()


def plot_combined(results, output_path):
    """
    Create a combined plot with both MSE and cosine similarity.
    
    Args:
        results: list of result dicts from experiments
        output_path: path to save the plot
    """
    num_layers_list = [r['num_layers'] for r in results]
    
    lsa_means = np.array([r.get('mse_lsa_mean', np.nan) for r in results], dtype=float)
    lsa_stds = np.array([r.get('mse_lsa_std', np.nan) for r in results], dtype=float)
    gd_means = np.array([r.get('mse_gd_mean', np.nan) for r in results], dtype=float)
    gd_stds = np.array([r.get('mse_gd_std', np.nan) for r in results], dtype=float)
    softmax_means = np.array([r.get('mse_softmax_mean', np.nan) for r in results], dtype=float)
    softmax_stds = np.array([r.get('mse_softmax_std', np.nan) for r in results], dtype=float)
    kernel_means = np.array([r.get('mse_kernel_mean', np.nan) for r in results], dtype=float)
    kernel_stds = np.array([r.get('mse_kernel_std', np.nan) for r in results], dtype=float)

    cos_lsa_means = np.array([r.get('cosine_sim_lsa_mean', r.get('cosine_sim_mean', np.nan)) for r in results], dtype=float)
    cos_lsa_stds = np.array([r.get('cosine_sim_lsa_std', r.get('cosine_sim_std', np.nan)) for r in results], dtype=float)
    cos_softmax_means = np.array([r.get('cosine_sim_softmax_mean', np.nan) for r in results], dtype=float)
    cos_softmax_stds = np.array([r.get('cosine_sim_softmax_std', np.nan) for r in results], dtype=float)
    cos_kernel_means = np.array([r.get('cosine_sim_kernel_mean', np.nan) for r in results], dtype=float)
    cos_kernel_stds = np.array([r.get('cosine_sim_kernel_std', np.nan) for r in results], dtype=float)

    lowrank_keys = _collect_lowrank_keys(results, "mse_")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: MSE comparison
    if np.isfinite(lsa_means).any():
        ax1.errorbar(num_layers_list, lsa_means, yerr=lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA', color='#4472C4')
    if np.isfinite(gd_means).any():
        ax1.errorbar(num_layers_list, gd_means, yerr=gd_stds,
                    marker='s', linewidth=2, markersize=8, capsize=5,
                    label='T-step GD', color='#ED7D31')
    if np.isfinite(softmax_means).any():
        ax1.errorbar(num_layers_list, softmax_means, yerr=softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax', color='#5B9BD5')
    markers = ['v', 'P', 'X', '*']
    for i, sk in enumerate(lowrank_keys):
        mse_vals, mse_std = _extract_series(results, f"mse_{sk}")
        ax1.errorbar(num_layers_list, mse_vals, yerr=mse_std,
                    marker=markers[i % len(markers)], linewidth=2, markersize=8, capsize=5,
                    label=sk.replace("lowrank_", "Low-rank "), color='#C00000')
    if np.isfinite(kernel_means).any():
        ax1.errorbar(num_layers_list, kernel_means, yerr=kernel_stds,
                    marker='D', linewidth=2, markersize=8, capsize=5,
                    label='Kernelized Linear', color='#70AD47')
    ax1.set_xlabel('Number of Layers / GD Steps', fontsize=14)
    ax1.set_ylabel('Mean Squared Error', fontsize=14)
    ax1.set_title('MSE Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(num_layers_list)
    
    # Plot 2: Cosine similarity
    if np.isfinite(cos_lsa_means).any():
        ax2.errorbar(num_layers_list, cos_lsa_means, yerr=cos_lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA vs GD', color='#4472C4')
    if np.isfinite(cos_softmax_means).any():
        ax2.errorbar(num_layers_list, cos_softmax_means, yerr=cos_softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax vs GD', color='#5B9BD5')
    for i, sk in enumerate(lowrank_keys):
        cos_vals, cos_std = _extract_series(results, f"cosine_sim_{sk}")
        ax2.errorbar(num_layers_list, cos_vals, yerr=cos_std,
                    marker=markers[i % len(markers)], linewidth=2, markersize=8, capsize=5,
                    label=f"{sk.replace('lowrank_', 'Low-rank ')} vs GD", color='#C00000')
    if np.isfinite(cos_kernel_means).any():
        ax2.errorbar(num_layers_list, cos_kernel_means, yerr=cos_kernel_stds,
                    marker='D', linewidth=2, markersize=8, capsize=5,
                    label='Kernelized Linear vs GD', color='#70AD47')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Alignment')
    ax2.set_xlabel('Number of Layers / GD Steps', fontsize=14)
    ax2.set_ylabel('Cosine Similarity', fontsize=14)
    ax2.set_title('Weight Update Alignment vs GD', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(num_layers_list)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")
    plt.close()


def save_results_table(results, output_path):
    """
    Save results as a CSV table.
    
    Args:
        results: list of result dicts from experiments
        output_path: path to save the CSV file
    """
    headers = [
        "num_layers",
        "mse_gd_mean", "mse_gd_std",
        "mse_lsa_mean", "mse_lsa_std",
        "mse_softmax_mean", "mse_softmax_std",
        "mse_kernel_mean", "mse_kernel_std",
        "cosine_sim_lsa_mean", "cosine_sim_lsa_std",
        "cosine_sim_softmax_mean", "cosine_sim_softmax_std",
        "cosine_sim_kernel_mean", "cosine_sim_kernel_std",
    ]

    lowrank_keys = set()
    for r in results:
        for k in r.keys():
            if k.startswith("mse_lowrank_") and k.endswith("_mean"):
                lowrank_keys.add(k.replace("_mean", ""))
    lowrank_keys = sorted(lowrank_keys)
    for base in lowrank_keys:
        headers.append(f"{base}_mean")
        headers.append(f"{base}_std")
        headers.append(f"{base.replace('mse_', 'cosine_sim_')}_mean")
        headers.append(f"{base.replace('mse_', 'cosine_sim_')}_std")

    def _get(r, key, fallback_key=None):
        if key in r:
            return r[key]
        if fallback_key and fallback_key in r:
            return r[fallback_key]
        return float('nan')

    with open(output_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = []
            for h in headers:
                row.append(_get(r, h))
            f.write(",".join([f"{v}" for v in row]) + "\n")
    
    print(f"Saved results table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot multi-layer LSA experiment results')
    parser.add_argument('--results_file', type=str, 
                        default='./results/lsa_multilayer/all_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='./results/lsa_multilayer',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run lsa_gd_multilayer.py first to generate results.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results for {len(results)} experiments")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # MSE comparison
    mse_path = output_dir / 'lsa_vs_gd_mse.png'
    plot_mse_comparison(results, mse_path)
    
    # Cosine similarity
    cosine_path = output_dir / 'lsa_vs_gd_cosine.png'
    plot_cosine_similarity(results, cosine_path)
    
    # Combined plot
    combined_path = output_dir / 'lsa_vs_gd_combined.png'
    plot_combined(results, combined_path)
    
    # Save table
    table_path = output_dir / 'results_table.csv'
    save_results_table(results, table_path)
    
    print("\nAll plots generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
