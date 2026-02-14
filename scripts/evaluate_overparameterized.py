#!/usr/bin/env python3
"""
Evaluate transformer vs gradient descent (least squares) for overparameterized case.
This compares the transformer's learned in-context algorithm against the optimal 
closed-form solution when we have more examples than dimensions (n_points > n_dims).
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
from src.data.tasks import LinearRegressionTask, mean_squared_error
from src.models.transformer import InContextTransformer
from src.evaluation.baselines import least_squares_solution, predict_from_weights


def evaluate_on_prompts(model, data_sampler, n_dims, n_points_list, n_eval=1000, device='cpu'):
    """
    Evaluate model and baseline on test prompts with varying context lengths.
    
    Args:
        model: Trained transformer model
        data_sampler: Data sampler for generating test data
        n_dims: Dimension of the problem
        n_points_list: List of context lengths to evaluate
        n_eval: Number of test prompts per context length
        device: Device to run on
    
    Returns:
        results: Dictionary with errors for each method
    """
    model.eval()
    
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
                # Combine train + test for in-context learning
                xs_full = torch.cat([xs_train, xs_test], dim=1)
                ys_full = torch.cat([ys_train, ys_test], dim=1)
                
                # Predict at the last position (test point)
                y_pred_transformer = model(xs_full, ys_full, inds=[-1])
                transformer_error = ((y_pred_transformer - ys_test) ** 2).mean().item()
                transformer_errors[n_points].append(transformer_error)
                
                # Least squares baseline (only if n_points >= n_dims for overparameterized)
                if n_points >= n_dims:
                    w_hat = least_squares_solution(xs_train, ys_train)
                    y_pred_ls = predict_from_weights(xs_test, w_hat)
                    ls_error = ((y_pred_ls - ys_test) ** 2).mean().item()
                    least_squares_errors[n_points].append(ls_error)
    
    # Compute mean and std errors
    results = {
        'transformer': {
            'mean': [np.mean(transformer_errors[n]) for n in n_points_list],
            'std': [np.std(transformer_errors[n]) for n in n_points_list],
        },
        'least_squares': {
            'mean': [np.mean(least_squares_errors[n]) if n >= n_dims else np.nan 
                    for n in n_points_list],
            'std': [np.std(least_squares_errors[n]) if n >= n_dims else np.nan 
                   for n in n_points_list],
        },
        'n_points_list': n_points_list,
        'n_dims': n_dims
    }
    
    return results


def plot_results(results, output_path):
    """
    Plot comparison between transformer and least squares.
    
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
    ax.plot(n_points_list, transformer_mean, 'o-', label='Transformer', linewidth=2)
    ax.fill_between(
        n_points_list,
        np.array(transformer_mean) - np.array(transformer_std),
        np.array(transformer_mean) + np.array(transformer_std),
        alpha=0.2
    )
    
    # Plot least squares (only for overparameterized regime)
    ls_mean = results['least_squares']['mean']
    ls_std = results['least_squares']['std']
    
    # Filter out NaN values
    valid_indices = [i for i, n in enumerate(n_points_list) if n >= n_dims]
    valid_n_points = [n_points_list[i] for i in valid_indices]
    valid_ls_mean = [ls_mean[i] for i in valid_indices]
    valid_ls_std = [ls_std[i] for i in valid_indices]
    
    ax.plot(valid_n_points, valid_ls_mean, 's--', label='Least Squares (Optimal)', 
            linewidth=2, markersize=8)
    ax.fill_between(
        valid_n_points,
        np.array(valid_ls_mean) - np.array(valid_ls_std),
        np.array(valid_ls_mean) + np.array(valid_ls_std),
        alpha=0.2
    )
    
    # Add vertical line at n_dims
    ax.axvline(x=n_dims, color='gray', linestyle=':', alpha=0.7, 
              label=f'n_dims = {n_dims}')
    
    # Formatting
    ax.set_xlabel('Number of In-Context Examples', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(f'Transformer vs Optimal Solution ({n_dims}D Linear Regression)', 
                fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotation for overparameterized region
    ax.text(n_dims + 2, ax.get_ylim()[1] * 0.5, 'Overparameterized\n(n > d)', 
           fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate overparameterized performance')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file used for training')
    parser.add_argument('--n_eval', type=int, default=1000,
                       help='Number of test prompts per context length')
    parser.add_argument('--output', type=str, default='results/overparameterized_comparison.png',
                       help='Output path for plot')
    parser.add_argument('--min_points', type=int, default=5,
                       help='Minimum number of in-context examples')
    parser.add_argument('--max_points', type=int, default=40,
                       help='Maximum number of in-context examples')
    parser.add_argument('--step_points', type=int, default=5,
                       help='Step size for number of points')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    n_dims = config['n_dims']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {args.checkpoint}")
    print(f"Using device: {device}")
    print(f"Problem dimension: {n_dims}")
    
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
    
    # Generate list of context lengths to evaluate
    n_points_list = list(range(args.min_points, args.max_points + 1, args.step_points))
    print(f"Evaluating context lengths: {n_points_list}")
    
    # Run evaluation
    print(f"Running evaluation with {args.n_eval} test prompts per context length...")
    results = evaluate_on_prompts(
        model=model,
        data_sampler=data_sampler,
        n_dims=n_dims,
        n_points_list=n_points_list,
        n_eval=args.n_eval,
        device=device
    )
    
    # Save results
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_json_path = output_dir / 'overparameterized_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_json_path}")
    
    # Plot results
    plot_results(results, args.output)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Dimension: {n_dims}")
    print(f"\nTransformer errors (first few context lengths):")
    for i, n in enumerate(n_points_list[:5]):
        print(f"  {n:2d} examples: {results['transformer']['mean'][i]:.6f} ± {results['transformer']['std'][i]:.6f}")
    
    print(f"\nLeast Squares errors (overparameterized regime):")
    for i, n in enumerate(n_points_list):
        if n >= n_dims:
            print(f"  {n:2d} examples: {results['least_squares']['mean'][i]:.6f} ± {results['least_squares']['std'][i]:.6f}")


if __name__ == '__main__':
    main()
