#!/usr/bin/env python3
"""
Compare transformer performance with gradient descent (least squares) 
in the overparameterized regime (n > d).

This script specifically focuses on demonstrating that transformers learn
to match the optimal closed-form solution when sufficient context is provided.
"""
import os
import sys
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.samplers import GaussianSampler
from src.data.tasks import LinearRegressionTask
from src.models.transformer import InContextTransformer
from src.evaluation.baselines import least_squares_solution, predict_from_weights


def evaluate_overparameterized_regime(model, data_sampler, n_dims, n_points_range, n_eval=1000, device='cpu'):
    """
    Evaluate transformer vs least squares in overparameterized regime only.
    
    Args:
        model: Trained transformer model
        data_sampler: Data sampler for generating test data
        n_dims: Dimension of the problem
        n_points_range: Tuple (min_points, max_points) where min_points >= n_dims
        n_eval: Number of test prompts per context length
        device: Device to run on
    
    Returns:
        results: Dictionary with errors for both methods
    """
    model.eval()
    
    min_points, max_points = n_points_range
    # Only evaluate in overparameterized regime
    n_points_list = list(range(max(min_points, n_dims), max_points + 1))
    
    print(f"\nEvaluating overparameterized regime (n >= {n_dims})")
    print(f"Context lengths: {n_points_list[0]} to {n_points_list[-1]}")
    
    transformer_errors = {n: [] for n in n_points_list}
    least_squares_errors = {n: [] for n in n_points_list}
    
    with torch.no_grad():
        for n_points in tqdm(n_points_list, desc="Context lengths"):
            for _ in range(n_eval):
                # Sample a single task
                task = LinearRegressionTask(
                    n_dims=n_dims,
                    batch_size=1,
                    n_dims_truncated=n_dims,
                    device=device
                )
                
                # Sample training points
                xs_train = data_sampler.sample_xs(
                    n_points=n_points,
                    batch_size=1,
                    n_dims_truncated=n_dims,
                    device=device
                )
                ys_train = task.evaluate(xs_train)
                
                # Sample test point
                xs_test = data_sampler.sample_xs(
                    n_points=1,
                    batch_size=1,
                    n_dims_truncated=n_dims,
                    device=device
                )
                ys_test = task.evaluate(xs_test)
                
                # Transformer prediction
                # Concatenate train and test, but use dummy value for test y
                xs_full = torch.cat([xs_train, xs_test], dim=1)
                # Use zeros as placeholder for test y (model shouldn't see it)
                ys_dummy = torch.zeros_like(ys_test)
                ys_full = torch.cat([ys_train, ys_dummy], dim=1)
                y_pred_transformer = model(xs_full, ys_full, inds=[-1])
                transformer_error = ((y_pred_transformer - ys_test) ** 2).item()
                transformer_errors[n_points].append(transformer_error)
                
                # Least squares (optimal solution)
                w_hat = least_squares_solution(xs_train, ys_train)
                y_pred_ls = predict_from_weights(xs_test, w_hat)
                ls_error = ((y_pred_ls - ys_test) ** 2).item()
                least_squares_errors[n_points].append(ls_error)
    
    # Compute statistics
    results = {
        'transformer': {
            'mean': [np.mean(transformer_errors[n]) for n in n_points_list],
            'std': [np.std(transformer_errors[n]) for n in n_points_list],
            'median': [np.median(transformer_errors[n]) for n in n_points_list],
        },
        'least_squares': {
            'mean': [np.mean(least_squares_errors[n]) for n in n_points_list],
            'std': [np.std(least_squares_errors[n]) for n in n_points_list],
            'median': [np.median(least_squares_errors[n]) for n in n_points_list],
        },
        'n_points_list': n_points_list,
        'n_dims': n_dims
    }
    
    return results


def plot_overparameterized_comparison(results, output_path):
    """
    Create a focused plot showing transformer matching gradient descent.
    
    Args:
        results: Dictionary with evaluation results
        output_path: Path to save the plot
    """
    n_points_list = results['n_points_list']
    n_dims = results['n_dims']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot transformer
    transformer_mean = results['transformer']['mean']
    transformer_std = results['transformer']['std']
    ax.plot(n_points_list, transformer_mean, 'o-', 
           label='Transformer (In-Context Learning)', 
           linewidth=2.5, markersize=7, color='#4472C4')
    ax.fill_between(
        n_points_list,
        np.array(transformer_mean) - np.array(transformer_std),
        np.array(transformer_mean) + np.array(transformer_std),
        alpha=0.2,
        color='#4472C4'
    )
    
    # Plot least squares
    ls_mean = results['least_squares']['mean']
    ls_std = results['least_squares']['std']
    ax.plot(n_points_list, ls_mean, 's--', 
           label='Least Squares (Optimal)', 
           linewidth=2.5, markersize=7, color='#ED7D31')
    ax.fill_between(
        n_points_list,
        np.array(ls_mean) - np.array(ls_std),
        np.array(ls_mean) + np.array(ls_std),
        alpha=0.2,
        color='#ED7D31'
    )
    
    # Add vertical line at n_dims to mark the boundary
    ax.axvline(x=n_dims, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    ax.text(n_dims + 0.5, ax.get_ylim()[1] * 0.9, 
           f'n = d = {n_dims}\n(Boundary)', 
           fontsize=10, alpha=0.7, va='top')
    
    # Formatting
    ax.set_xlabel('Number of In-Context Examples (n)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Squared Error', fontsize=13, fontweight='bold')
    ax.set_title(f'Transformer Matches Gradient Descent in Overparameterized Regime\n({n_dims}D Linear Regression)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(max(transformer_mean), max(ls_mean)) * 1.2)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    
    # Add annotation box
    textstr = f'Overparameterized Regime (n > d)\n' \
              f'Transformer learns optimal solution\n' \
              f'Final error: {transformer_mean[-1]:.4f} ≈ {ls_mean[-1]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare transformer with gradient descent in overparameterized regime'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file used for training')
    parser.add_argument('--n_eval', type=int, default=1000,
                       help='Number of test prompts per context length')
    parser.add_argument('--output', type=str, 
                       default='results/overparameterized_match.png',
                       help='Output path for plot')
    parser.add_argument('--min_points', type=int, default=None,
                       help='Minimum context length (defaults to n_dims)')
    parser.add_argument('--max_points', type=int, default=40,
                       help='Maximum context length')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    n_dims = config['n_dims']
    min_points = args.min_points if args.min_points else n_dims
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Overparameterized Regime Evaluation")
    print("="*60)
    print(f"Loading model from: {args.checkpoint}")
    print(f"Using device: {device}")
    print(f"Problem dimension: {n_dims}")
    print(f"Overparameterized range: {min_points} to {args.max_points} examples")
    
    # Initialize model
    model = InContextTransformer(
        n_dims=n_dims,
        n_positions=config['max_n_points'],
        n_embd=config['n_embd'],
        n_layer=config['n_layer'],
        n_head=config['n_head']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    
    # Initialize data sampler
    data_sampler = GaussianSampler(n_dims=n_dims)
    
    # Run evaluation
    print(f"\nEvaluating with {args.n_eval} test prompts per context length...")
    results = evaluate_overparameterized_regime(
        model=model,
        data_sampler=data_sampler,
        n_dims=n_dims,
        n_points_range=(min_points, args.max_points),
        n_eval=args.n_eval,
        device=device
    )
    
    # Save results
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_json_path = output_dir / 'overparameterized_match_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_json_path}")
    
    # Plot results
    plot_overparameterized_comparison(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Transformer vs Gradient Descent (Overparameterized)")
    print("="*60)
    n_points_list = results['n_points_list']
    
    print(f"\n{'Examples':<12} {'Transformer':<20} {'Least Squares':<20} {'Match?':<10}")
    print("-" * 60)
    
    for i, n in enumerate(n_points_list[::5]):  # Show every 5th point
        idx = n_points_list.index(n)
        t_err = results['transformer']['mean'][idx]
        ls_err = results['least_squares']['mean'][idx]
        ratio = t_err / ls_err if ls_err > 0 else 0
        match = "✓" if 0.8 < ratio < 1.2 else "✗"
        print(f"{n:<12} {t_err:<20.6f} {ls_err:<20.6f} {match:<10}")
    
    # Final comparison
    final_t = results['transformer']['mean'][-1]
    final_ls = results['least_squares']['mean'][-1]
    final_ratio = final_t / final_ls if final_ls > 0 else 0
    
    print("\n" + "="*60)
    print(f"Final Performance (at {n_points_list[-1]} examples):")
    print(f"  Transformer:    {final_t:.6f}")
    print(f"  Least Squares:  {final_ls:.6f}")
    print(f"  Ratio (T/LS):   {final_ratio:.3f}")
    
    if 0.9 < final_ratio < 1.1:
        print("\n✓ CONFIRMED: Transformer matches gradient descent performance!")
    else:
        print("\n⚠ Warning: Performance gap detected. May need more training.")
    print("="*60)


if __name__ == '__main__':
    main()
