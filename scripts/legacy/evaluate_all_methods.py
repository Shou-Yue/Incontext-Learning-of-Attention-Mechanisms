#!/usr/bin/env python3
"""
Evaluate transformer against all baseline methods from the paper:
- Transformer (learned in-context)
- Least Squares (optimal closed-form)
- 3-Nearest Neighbors
- Averaging (simple mean baseline)

Reproduces Figure 2 from the paper.
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
from src.evaluation.baselines import (
    least_squares_solution,
    ridge_regression_solution, 
    predict_from_weights,
    knn_prediction,
    averaging_prediction
)


def evaluate_all_methods(model, data_sampler, n_dims, n_points_list, n_eval=1000, device='cpu'):
    """
    Evaluate all methods on test prompts with varying context lengths.
    
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
    
    # Store errors for each method
    errors = {
        'transformer': {n: [] for n in n_points_list},
        'least_squares': {n: [] for n in n_points_list},
        'knn': {n: [] for n in n_points_list},
        'averaging': {n: [] for n in n_points_list}
    }
    
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
                
                # 1. Transformer prediction
                # Concatenate train and test, but use dummy value for test y
                xs_full = torch.cat([xs_train, xs_test], dim=1)
                # Use zeros as placeholder for test y (model shouldn't see it)
                ys_dummy = torch.zeros_like(ys_test)
                ys_full = torch.cat([ys_train, ys_dummy], dim=1)
                y_pred_transformer = model(xs_full, ys_full, inds=[-1])
                transformer_error = ((y_pred_transformer - ys_test) ** 2).item()
                errors['transformer'][n_points].append(transformer_error)
                
                # 2. Least Squares / Ridge Regression
                # Use ridge regression for underparameterized (n < d) to avoid singularity
                # Use regular least squares for overparameterized (n >= d)
                if n_points >= n_dims:
                    w_hat = least_squares_solution(xs_train, ys_train)
                else:
                    # Small regularization for underparameterized case
                    w_hat = ridge_regression_solution(xs_train, ys_train, lambda_reg=0.001)
                y_pred_ls = predict_from_weights(xs_test, w_hat)
                ls_error = ((y_pred_ls - ys_test) ** 2).item()
                errors['least_squares'][n_points].append(ls_error)
                
                # 3. k-Nearest Neighbors
                if n_points >= 3:  # Need at least 3 points for 3-NN
                    y_pred_knn = knn_prediction(xs_train, ys_train, xs_test, k=3)
                    knn_error = ((y_pred_knn - ys_test) ** 2).item()
                    errors['knn'][n_points].append(knn_error)
                
                # 4. Averaging
                y_pred_avg = averaging_prediction(xs_train, ys_train, xs_test)
                avg_error = ((y_pred_avg - ys_test) ** 2).item()
                errors['averaging'][n_points].append(avg_error)
    
    # Compute mean and std errors
    results = {
        'transformer': {
            'mean': [np.mean(errors['transformer'][n]) for n in n_points_list],
            'std': [np.std(errors['transformer'][n]) for n in n_points_list],
        },
        'least_squares': {
            'mean': [np.mean(errors['least_squares'][n]) for n in n_points_list],
            'std': [np.std(errors['least_squares'][n]) for n in n_points_list],
        },
        'knn': {
            'mean': [np.mean(errors['knn'][n]) if n >= 3 and len(errors['knn'][n]) > 0 
                    else np.nan for n in n_points_list],
            'std': [np.std(errors['knn'][n]) if n >= 3 and len(errors['knn'][n]) > 0 
                   else np.nan for n in n_points_list],
        },
        'averaging': {
            'mean': [np.mean(errors['averaging'][n]) for n in n_points_list],
            'std': [np.std(errors['averaging'][n]) for n in n_points_list],
        },
        'n_points_list': n_points_list,
        'n_dims': n_dims
    }
    
    return results


def plot_comparison(results, output_path):
    """
    Plot comparison between all methods (reproducing paper's Figure 2).
    
    Args:
        results: Dictionary with evaluation results
        output_path: Path to save the plot
    """
    n_points_list = results['n_points_list']
    n_dims = results['n_dims']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors matching the paper
    colors = {
        'transformer': '#4472C4',      # Blue
        'least_squares': '#ED7D31',    # Orange
        'knn': '#70AD47',              # Green
        'averaging': '#C55A11'         # Dark orange
    }
    
    methods = [
        ('transformer', 'Transformer', 'o-'),
        ('least_squares', 'Least Squares', 's-'),
        ('knn', '3-Nearest Neighbors', '^-'),
        ('averaging', 'Averaging', 'd-')
    ]
    
    for method_key, method_label, marker in methods:
        mean_vals = results[method_key]['mean']
        std_vals = results[method_key]['std']
        
        # Filter out NaN values
        valid_indices = [i for i, val in enumerate(mean_vals) if not np.isnan(val)]
        if not valid_indices:
            continue
            
        valid_n_points = [n_points_list[i] for i in valid_indices]
        valid_mean = [mean_vals[i] for i in valid_indices]
        valid_std = [std_vals[i] for i in valid_indices]
        
        # Plot mean with error bands
        ax.plot(valid_n_points, valid_mean, marker, 
               label=method_label, linewidth=2, markersize=6,
               color=colors[method_key])
        ax.fill_between(
            valid_n_points,
            np.array(valid_mean) - np.array(valid_std),
            np.array(valid_mean) + np.array(valid_std),
            alpha=0.15,
            color=colors[method_key]
        )
    
    # Add horizontal line at y=1 (as in paper)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add vertical line at n_dims
    ax.axvline(x=n_dims, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(n_dims + 1, ax.get_ylim()[1] * 0.8, f'n_dims={n_dims}', 
           fontsize=9, alpha=0.7)
    
    # Formatting to match paper
    ax.set_xlabel('Number of in-context examples', fontsize=12)
    ax.set_ylabel('Squared error', fontsize=12)
    ax.set_title(f'Linear Functions ({n_dims}D)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.2)  # Focus on 0-1.2 range
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, framealpha=0.95)
    
    # Set x-axis limits starting from 0
    ax.set_xlim(0, max(n_points_list) + 2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all methods (reproduce paper Figure 2)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file used for training')
    parser.add_argument('--n_eval', type=int, default=1000,
                       help='Number of test prompts per context length')
    parser.add_argument('--output', type=str, default='results/all_methods_comparison.png',
                       help='Output path for plot')
    parser.add_argument('--max_points', type=int, default=40,
                       help='Maximum number of in-context examples')
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
    
    # Generate list of context lengths from 1 to max_points
    n_points_list = list(range(1, args.max_points + 1))
    print(f"Evaluating context lengths: 1 to {args.max_points} (total: {len(n_points_list)} points)")
    
    # Run evaluation
    print(f"Running evaluation with {args.n_eval} test prompts per context length...")
    results = evaluate_all_methods(
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
    
    results_json_path = output_dir / 'all_methods_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_json_path}")
    
    # Plot results
    plot_comparison(results, args.output)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Dimension: {n_dims}\n")
    
    for method in ['transformer', 'least_squares', 'knn', 'averaging']:
        print(f"{method.replace('_', ' ').title()}:")
        for i, n in enumerate(n_points_list):
            mean_val = results[method]['mean'][i]
            std_val = results[method]['std'][i]
            if not np.isnan(mean_val):
                print(f"  {n:2d} examples: {mean_val:.6f} ± {std_val:.6f}")
        print()


if __name__ == '__main__':
    main()
