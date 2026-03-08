#!/usr/bin/env python3
"""
Evaluate on 7D with up to 15 points (matching current training curriculum).
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

# HARDCODED FOR CURRENT TRAINING STATE
N_DIMS = 7
MAX_POINTS = 15

def evaluate_current_curriculum(model, n_eval=1000, device='cpu'):
    """Evaluate on current curriculum level (7D, up to 15 points)."""
    model.eval()
    
    data_sampler = GaussianSampler(n_dims=N_DIMS)
    n_points_list = list(range(1, MAX_POINTS + 1))
    
    errors = {
        'transformer': {n: [] for n in n_points_list},
        'least_squares': {n: [] for n in n_points_list},
        'knn': {n: [] for n in n_points_list},
        'averaging': {n: [] for n in n_points_list}
    }
    
    print(f"Evaluating on {N_DIMS}D with {n_points_list[0]}-{n_points_list[-1]} points")
    
    with torch.no_grad():
        for n_points in tqdm(n_points_list, desc="Context lengths"):
            for _ in range(n_eval):
                task = LinearRegressionTask(
                    n_dims=N_DIMS,
                    batch_size=1,
                    n_dims_truncated=N_DIMS,
                    device=device
                )
                
                xs_train = data_sampler.sample_xs(
                    n_points=n_points,
                    batch_size=1,
                    n_dims_truncated=N_DIMS,
                    device=device
                )
                ys_train = task.evaluate(xs_train)
                
                xs_test = data_sampler.sample_xs(
                    n_points=1,
                    batch_size=1,
                    n_dims_truncated=N_DIMS,
                    device=device
                )
                ys_test = task.evaluate(xs_test)
                
                # Transformer
                xs_full = torch.cat([xs_train, xs_test], dim=1)
                ys_dummy = torch.zeros_like(ys_test)
                ys_full = torch.cat([ys_train, ys_dummy], dim=1)
                y_pred_transformer = model(xs_full, ys_full, inds=[-1])
                transformer_error = ((y_pred_transformer - ys_test) ** 2).item()
                errors['transformer'][n_points].append(transformer_error)
                
                # Least Squares
                if n_points >= N_DIMS:
                    w_hat = least_squares_solution(xs_train, ys_train)
                else:
                    w_hat = ridge_regression_solution(xs_train, ys_train, lambda_reg=0.001)
                y_pred_ls = predict_from_weights(xs_test, w_hat)
                ls_error = ((y_pred_ls - ys_test) ** 2).item()
                errors['least_squares'][n_points].append(ls_error)
                
                # k-NN
                if n_points >= 3:
                    y_pred_knn = knn_prediction(xs_train, ys_train, xs_test, k=3)
                    knn_error = ((y_pred_knn - ys_test) ** 2).item()
                    errors['knn'][n_points].append(knn_error)
                
                # Averaging
                y_pred_avg = averaging_prediction(xs_train, ys_train, xs_test)
                avg_error = ((y_pred_avg - ys_test) ** 2).item()
                errors['averaging'][n_points].append(avg_error)
    
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
            'mean': [np.mean(errors['knn'][n]) if n >= 3 else np.nan for n in n_points_list],
            'std': [np.std(errors['knn'][n]) if n >= 3 else np.nan for n in n_points_list],
        },
        'averaging': {
            'mean': [np.mean(errors['averaging'][n]) for n in n_points_list],
            'std': [np.std(errors['averaging'][n]) for n in n_points_list],
        },
        'n_points_list': n_points_list,
        'n_dims': N_DIMS
    }
    
    return results

def plot_results(results, output_path):
    """Plot results."""
    n_points_list = results['n_points_list']
    n_dims = results['n_dims']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'transformer': '#4472C4',
        'least_squares': '#ED7D31',
        'knn': '#70AD47',
        'averaging': '#C55A11'
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
        
        valid_indices = [i for i, val in enumerate(mean_vals) if not np.isnan(val)]
        if not valid_indices:
            continue
            
        valid_n_points = [n_points_list[i] for i in valid_indices]
        valid_mean = [mean_vals[i] for i in valid_indices]
        valid_std = [std_vals[i] for i in valid_indices]
        
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
    
    ax.axvline(x=n_dims, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(n_dims + 0.3, ax.get_ylim()[1] * 0.9, f'n=d={n_dims}', fontsize=10)
    
    ax.set_xlabel('Number of In-Context Examples', fontsize=12)
    ax.set_ylabel('Squared Error', fontsize=12)
    ax.set_title(f'Linear Functions ({n_dims}D) - Current Training Level', fontsize=14, fontweight='bold')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

def main():
    checkpoint_path = 'checkpoints/test_local/checkpoint_latest.pt'
    config_path = 'configs/test_local.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with ORIGINAL config dimensions
    model = InContextTransformer(
        n_dims=config['n_dims'],  # Use original 20
        n_positions=config['max_n_points'],
        n_embd=config['n_embd'],
        n_layer=config['n_layer'],
        n_head=config['n_head']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    print(f"Evaluating on {N_DIMS}D (current curriculum level)")
    
    results = evaluate_current_curriculum(model, n_eval=1000, device=device)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/7d_15p_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_results(results, 'results/7d_15p_comparison.png')
    
    # Print summary
    print("\n=== Summary ===")
    for method in ['transformer', 'least_squares']:
        print(f"\n{method.replace('_', ' ').title()}:")
        for i, n in enumerate(results['n_points_list']):
            mean = results[method]['mean'][i]
            std = results[method]['std'][i]
            if not np.isnan(mean):
                print(f"  {n:2d} examples: {mean:.6f} ± {std:.6f}")

if __name__ == '__main__':
    main()
