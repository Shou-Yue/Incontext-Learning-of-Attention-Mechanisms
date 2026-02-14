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

def plot_mse_comparison(results, output_path):
    """
    Plot MSE comparison between LSA and GD across different num_layers.
    
    Args:
        results: list of result dicts from experiments
        output_path: path to save the plot
    """
    num_layers_list = [r['num_layers'] for r in results]
    
    lsa_means = [r.get('mse_lsa_mean') for r in results]
    lsa_stds = [r.get('mse_lsa_std') for r in results]
    gd_means = [r.get('mse_gd_mean') for r in results]
    gd_stds = [r.get('mse_gd_std') for r in results]

    softmax_means = [r.get('mse_softmax_mean') for r in results]
    softmax_stds = [r.get('mse_softmax_std') for r in results]
    linformer_means = [r.get('mse_linformer_mean') for r in results]
    linformer_stds = [r.get('mse_linformer_std') for r in results]
    kernel_means = [r.get('mse_kernel_mean') for r in results]
    kernel_stds = [r.get('mse_kernel_std') for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot LSA
    if all(v is not None for v in lsa_means):
        ax.errorbar(num_layers_list, lsa_means, yerr=lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA', color='#4472C4')

    # Plot GD
    if all(v is not None for v in gd_means):
        ax.errorbar(num_layers_list, gd_means, yerr=gd_stds,
                    marker='s', linewidth=2, markersize=8, capsize=5,
                    label='T-step GD', color='#ED7D31')

    # Plot Softmax
    if all(v is not None for v in softmax_means):
        ax.errorbar(num_layers_list, softmax_means, yerr=softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax', color='#5B9BD5')

    # Plot Linformer (low-rank softmax)
    if all(v is not None for v in linformer_means):
        ax.errorbar(num_layers_list, linformer_means, yerr=linformer_stds,
                    marker='v', linewidth=2, markersize=8, capsize=5,
                    label='Low-rank Softmax', color='#C00000')

    # Plot Kernelized Linear
    if all(v is not None for v in kernel_means):
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
    
    cos_lsa_means = [r.get('cosine_sim_lsa_mean') or r.get('cosine_sim_mean') for r in results]
    cos_lsa_stds = [r.get('cosine_sim_lsa_std') or r.get('cosine_sim_std') for r in results]
    cos_softmax_means = [r.get('cosine_sim_softmax_mean') for r in results]
    cos_softmax_stds = [r.get('cosine_sim_softmax_std') for r in results]
    cos_linformer_means = [r.get('cosine_sim_linformer_mean') for r in results]
    cos_linformer_stds = [r.get('cosine_sim_linformer_std') for r in results]
    cos_kernel_means = [r.get('cosine_sim_kernel_mean') for r in results]
    cos_kernel_stds = [r.get('cosine_sim_kernel_std') for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cosine similarity (LSA baseline)
    if all(v is not None for v in cos_lsa_means):
        ax.errorbar(num_layers_list, cos_lsa_means, yerr=cos_lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA vs GD', color='#4472C4')

    if all(v is not None for v in cos_softmax_means):
        ax.errorbar(num_layers_list, cos_softmax_means, yerr=cos_softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax vs GD', color='#5B9BD5')

    if all(v is not None for v in cos_linformer_means):
        ax.errorbar(num_layers_list, cos_linformer_means, yerr=cos_linformer_stds,
                    marker='v', linewidth=2, markersize=8, capsize=5,
                    label='Low-rank Softmax vs GD', color='#C00000')

    if all(v is not None for v in cos_kernel_means):
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
    
    lsa_means = [r.get('mse_lsa_mean') for r in results]
    lsa_stds = [r.get('mse_lsa_std') for r in results]
    gd_means = [r.get('mse_gd_mean') for r in results]
    gd_stds = [r.get('mse_gd_std') for r in results]
    softmax_means = [r.get('mse_softmax_mean') for r in results]
    softmax_stds = [r.get('mse_softmax_std') for r in results]
    linformer_means = [r.get('mse_linformer_mean') for r in results]
    linformer_stds = [r.get('mse_linformer_std') for r in results]
    kernel_means = [r.get('mse_kernel_mean') for r in results]
    kernel_stds = [r.get('mse_kernel_std') for r in results]

    cos_lsa_means = [r.get('cosine_sim_lsa_mean') or r.get('cosine_sim_mean') for r in results]
    cos_lsa_stds = [r.get('cosine_sim_lsa_std') or r.get('cosine_sim_std') for r in results]
    cos_softmax_means = [r.get('cosine_sim_softmax_mean') for r in results]
    cos_softmax_stds = [r.get('cosine_sim_softmax_std') for r in results]
    cos_linformer_means = [r.get('cosine_sim_linformer_mean') for r in results]
    cos_linformer_stds = [r.get('cosine_sim_linformer_std') for r in results]
    cos_kernel_means = [r.get('cosine_sim_kernel_mean') for r in results]
    cos_kernel_stds = [r.get('cosine_sim_kernel_std') for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: MSE comparison
    if all(v is not None for v in lsa_means):
        ax1.errorbar(num_layers_list, lsa_means, yerr=lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA', color='#4472C4')
    if all(v is not None for v in gd_means):
        ax1.errorbar(num_layers_list, gd_means, yerr=gd_stds,
                    marker='s', linewidth=2, markersize=8, capsize=5,
                    label='T-step GD', color='#ED7D31')
    if all(v is not None for v in softmax_means):
        ax1.errorbar(num_layers_list, softmax_means, yerr=softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax', color='#5B9BD5')
    if all(v is not None for v in linformer_means):
        ax1.errorbar(num_layers_list, linformer_means, yerr=linformer_stds,
                    marker='v', linewidth=2, markersize=8, capsize=5,
                    label='Low-rank Softmax', color='#C00000')
    if all(v is not None for v in kernel_means):
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
    if all(v is not None for v in cos_lsa_means):
        ax2.errorbar(num_layers_list, cos_lsa_means, yerr=cos_lsa_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label='LSA vs GD', color='#4472C4')
    if all(v is not None for v in cos_softmax_means):
        ax2.errorbar(num_layers_list, cos_softmax_means, yerr=cos_softmax_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5,
                    label='Softmax vs GD', color='#5B9BD5')
    if all(v is not None for v in cos_linformer_means):
        ax2.errorbar(num_layers_list, cos_linformer_means, yerr=cos_linformer_stds,
                    marker='v', linewidth=2, markersize=8, capsize=5,
                    label='Low-rank Softmax vs GD', color='#C00000')
    if all(v is not None for v in cos_kernel_means):
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
        "mse_linformer_mean", "mse_linformer_std",
        "mse_kernel_mean", "mse_kernel_std",
        "cosine_sim_lsa_mean", "cosine_sim_lsa_std",
        "cosine_sim_softmax_mean", "cosine_sim_softmax_std",
        "cosine_sim_linformer_mean", "cosine_sim_linformer_std",
        "cosine_sim_kernel_mean", "cosine_sim_kernel_std",
    ]

    def _get(r, key, fallback_key=None):
        if key in r:
            return r[key]
        if fallback_key and fallback_key in r:
            return r[fallback_key]
        return float('nan')

    with open(output_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                r["num_layers"],
                _get(r, "mse_gd_mean"),
                _get(r, "mse_gd_std"),
                _get(r, "mse_lsa_mean"),
                _get(r, "mse_lsa_std"),
                _get(r, "mse_softmax_mean"),
                _get(r, "mse_softmax_std"),
                _get(r, "mse_linformer_mean"),
                _get(r, "mse_linformer_std"),
                _get(r, "mse_kernel_mean"),
                _get(r, "mse_kernel_std"),
                _get(r, "cosine_sim_lsa_mean", "cosine_sim_mean"),
                _get(r, "cosine_sim_lsa_std", "cosine_sim_std"),
                _get(r, "cosine_sim_softmax_mean"),
                _get(r, "cosine_sim_softmax_std"),
                _get(r, "cosine_sim_linformer_mean"),
                _get(r, "cosine_sim_linformer_std"),
                _get(r, "cosine_sim_kernel_mean"),
                _get(r, "cosine_sim_kernel_std"),
            ]

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
