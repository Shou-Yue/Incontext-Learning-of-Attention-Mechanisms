#!/usr/bin/env python3
"""
Multi-layer LSA = Multi-step GD Experiment

Trains and evaluates multi-layer Linear Self-Attention models to approximate
multi-step gradient descent on linear regression tasks.

Based on Oswald et al. (2023): "Transformers Learn In-Context by Gradient Descent"
"""
import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lsa import MultiLayerLSA
from src.models.attention_variants import MultiLayerAttentionModel
from src.evaluation.gd_baseline import (
    generate_linear_regression_task,
    gd_t_steps,
    compute_cosine_similarity
)


def train_model(model, d, n_points, num_tasks=10000, batch_size=64,
                num_epochs=10, lr=1e-3, sigma=0.0, device='cpu'):
    """
    Train attention model on linear regression tasks.
    
    Args:
        model: MultiLayerLSA model
        d: input dimension
        n_points: number of training points per task
        num_tasks: number of training tasks
        batch_size: batch size for training
        num_epochs: number of epochs
        lr: learning rate
        sigma: label noise level
        device: torch device
    
    Returns:
        model: trained model
        train_losses: list of training losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    num_batches = num_tasks // batch_size
    
    print(f"Training {model.num_layers}-layer {model.__class__.__name__} model...")
    print(f"  d={d}, n={n_points}, tasks={num_tasks}, batch_size={batch_size}")
    print(f"  epochs={num_epochs}, lr={lr}, sigma={sigma}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for _ in pbar:
            # Generate batch of tasks
            xs, ys, w_true = generate_linear_regression_task(
                d, n_points, batch_size=batch_size, sigma=sigma, device=device
            )
            
            # Generate query points
            query_x = torch.randn(batch_size, 1, d, device=device)
            query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            
            # Add noise to query labels if needed
            if sigma > 0:
                query_y = query_y + torch.randn_like(query_y) * sigma
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(xs, ys, query_x)
            loss = criterion(y_pred, query_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")
    
    return model, train_losses


def evaluate_models(models, d, n_points, num_eval_tasks=1000,
                    batch_size=100, eta=0.1, sigma=0.0, device='cpu'):
    """
    Evaluate multiple models on the same tasks and compare to GD baseline.

    Args:
        models: dict of name -> model (all with same num_layers)
    """
    for m in models.values():
        m.to(device)
        m.eval()

    mse_lists = {name: [] for name in models}
    cos_lists = {name: [] for name in models}
    mse_gd_list = []

    num_batches = num_eval_tasks // batch_size
    num_layers = next(iter(models.values())).num_layers

    print(f"\nEvaluating {num_layers}-layer models: {', '.join(models.keys())}")

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Evaluation"):
            xs, ys, w_true = generate_linear_regression_task(
                d, n_points, batch_size=batch_size, sigma=sigma, device=device
            )

            query_x = torch.randn(batch_size, 1, d, device=device)
            query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)

            if sigma > 0:
                query_y = query_y + torch.randn_like(query_y) * sigma

            # GD baseline (T = num_layers)
            y_pred_gd, w_gd = gd_t_steps(xs, ys, query_x, eta=eta, T=num_layers)
            mse_gd = ((y_pred_gd - query_y) ** 2).mean().item()
            mse_gd_list.append(mse_gd)

            for name, model in models.items():
                y_pred = model(xs, ys, query_x)
                mse = ((y_pred - query_y) ** 2).mean().item()
                mse_lists[name].append(mse)

                w_model = model.get_weight_update(xs, ys, query_x)
                cos = compute_cosine_similarity(w_model, w_gd).mean().item()
                cos_lists[name].append(cos)

    results = {
        'mse_gd_mean': np.mean(mse_gd_list),
        'mse_gd_std': np.std(mse_gd_list)
    }

    for name in models:
        results[f"mse_{name}_mean"] = np.mean(mse_lists[name])
        results[f"mse_{name}_std"] = np.std(mse_lists[name])
        results[f"cosine_sim_{name}_mean"] = np.mean(cos_lists[name])
        results[f"cosine_sim_{name}_std"] = np.std(cos_lists[name])

    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-layer LSA = Multi-step GD Experiment')
    parser.add_argument('--d', type=int, default=20, help='Input dimension')
    parser.add_argument('--num_layers_list', type=int, nargs='+', default=[1, 2, 4],
                        help='List of num_layers to experiment with')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension for LSA (default: d)')
    parser.add_argument('--num_train_tasks', type=int, default=10000,
                        help='Number of training tasks')
    parser.add_argument('--num_eval_tasks', type=int, default=1000,
                        help='Number of evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Learning rate for GD baseline')
    parser.add_argument('--sigma', type=float, default=0.0,
                        help='Label noise level (std dev)')
    parser.add_argument('--output_dir', type=str, default='./results/lsa_multilayer',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['lsa', 'softmax', 'linformer', 'kernel'],
                        help='Model types to run: lsa, softmax, linformer, kernel')
    parser.add_argument('--lowrank_k_ratio', type=float, default=0.5,
                        help='Linformer proj_k ratio relative to n_tokens (e.g., 1.0, 0.75, 0.5)')
    parser.add_argument('--lowrank_share_ef', action='store_true',
                        help='Share E and F projection matrices in low-rank attention')
    parser.add_argument('--lowrank_orth_init', action='store_true',
                        help='Use orthogonal init for E/F in low-rank attention')
    parser.add_argument('--lowrank_identity_proj', action='store_true',
                        help='Force identity E/F when ratio=1.0')
    parser.add_argument('--lowrank_freeze_proj', action='store_true',
                        help='Freeze E/F projection matrices in low-rank attention')
    parser.add_argument('--lowrank_exact_match', action='store_true',
                        help='For ratio=1.0, force exact softmax equivalence in low-rank path')
    parser.add_argument('--lowrank_block_size', type=int, default=None,
                        help='Use blockwise low-rank projection with this block size')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Calculate n_points = 2d + 1
    n_points = 2 * args.d + 1
    n_tokens = n_points + 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Run experiments for each num_layers
    for num_layers in args.num_layers_list:
        print(f"\n{'='*60}")
        print(f"Experiment: num_layers = {num_layers}")
        print(f"{'='*60}")

        # Build and train all requested model types
        models = {}
        train_losses_by_model = {}

        for model_type in args.model_types:
            if model_type == 'lsa':
                model = MultiLayerLSA(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim
                )
            elif model_type == 'softmax':
                model = MultiLayerAttentionModel(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim,
                    attn_type='softmax',
                    causal=True
                )
            elif model_type == 'linformer':
                proj_k = max(1, int(round(n_tokens * args.lowrank_k_ratio)))
                identity_proj = args.lowrank_identity_proj and abs(args.lowrank_k_ratio - 1.0) < 1e-9
                if args.lowrank_exact_match and abs(args.lowrank_k_ratio - 1.0) < 1e-9:
                    identity_proj = True
                    freeze_proj = True
                    normalize_qk = False
                    learnable_scale = False
                else:
                    freeze_proj = args.lowrank_freeze_proj
                    normalize_qk = True
                    learnable_scale = True

                model = MultiLayerAttentionModel(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim,
                    attn_type='linformer',
                    n_tokens=n_tokens,  # include query token
                    proj_k=proj_k,
                    lowrank_kwargs={
                        'share_ef': args.lowrank_share_ef,
                        'orth_init': args.lowrank_orth_init,
                        'identity_proj': identity_proj,
                        'freeze_proj': freeze_proj,
                        'normalize_qk': normalize_qk,
                        'learnable_scale': learnable_scale,
                        'block_size': args.lowrank_block_size
                    }
                )
            elif model_type == 'kernel':
                model = MultiLayerAttentionModel(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim,
                    attn_type='kernel'
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            print(f"\nModel ({model_type}) parameters: {sum(p.numel() for p in model.parameters()):,}")

            model, train_losses = train_model(
                model=model,
                d=args.d,
                n_points=n_points,
                num_tasks=args.num_train_tasks,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                sigma=args.sigma,
                device=device
            )

            models[model_type] = model
            train_losses_by_model[model_type] = train_losses

            # Save model checkpoint
            checkpoint_path = output_dir / f'{model_type}_{num_layers}layer_checkpoint.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_layers': num_layers,
                'd': args.d,
                'hidden_dim': args.hidden_dim,
                'train_losses': train_losses,
                'model_type': model_type
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Evaluate all models on shared tasks
        results = evaluate_models(
            models=models,
            d=args.d,
            n_points=n_points,
            num_eval_tasks=args.num_eval_tasks,
            batch_size=100,
            eta=args.eta,
            sigma=args.sigma,
            device=device
        )

        results['num_layers'] = num_layers
        for model_type, losses in train_losses_by_model.items():
            results[f"train_losses_{model_type}"] = losses

        all_results.append(results)
    
    # Save all results
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All experiments complete!")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Layers':<10} {'MSE GD':<20} {'MSE LSA':<20} {'MSE Softmax':<20} {'MSE Linformer':<20} {'MSE Kernel':<20}")
    print("-"*70)
    for res in all_results:
        print(f"{res['num_layers']:<10} "
              f"{res['mse_gd_mean']:.6f} ± {res['mse_gd_std']:.4f}    "
              f"{res.get('mse_lsa_mean', float('nan')):.6f} ± {res.get('mse_lsa_std', float('nan')):.4f}    "
              f"{res.get('mse_softmax_mean', float('nan')):.6f} ± {res.get('mse_softmax_std', float('nan')):.4f}    "
              f"{res.get('mse_linformer_mean', float('nan')):.6f} ± {res.get('mse_linformer_std', float('nan')):.4f}    "
              f"{res.get('mse_kernel_mean', float('nan')):.6f} ± {res.get('mse_kernel_std', float('nan')):.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
