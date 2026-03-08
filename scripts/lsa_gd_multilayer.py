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
from src.models.gla import MultiLayerGLA
from src.models.gqa import GQATransformer
from src.models.sparse_causal import SparseICLModel
from src.evaluation.gd_baseline import (
    generate_linear_regression_task,
    gd_t_steps,
    compute_cosine_similarity
)


def train_model(model, d, n_points, num_tasks=10000, batch_size=64,
                num_epochs=10, lr=1e-3, sigma=0.0, device='cpu', use_amp=False):
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
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    
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
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                y_pred = model(xs, ys, query_x)
                loss = criterion(y_pred, query_y)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            if use_amp and device.type == "cuda":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
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
    parser.add_argument('--num_train_tasks', type=int, default=64000,
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
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision (AMP) on CUDA')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile on models (PyTorch 2.x)')
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['lsa', 'softmax', 'linformer', 'kernel', 'gla', 'gqa', 'sparse'],
                        help='Model types to run: lsa, softmax, linformer, kernel, gla, gqa, sparse')
    parser.add_argument('--lowrank_k_ratio', type=float, default=0.5,
                        help='Single low-rank ratio (backward compatible). Ignored if --lowrank_k_ratios is set.')
    parser.add_argument('--lowrank_k_ratios', type=float, nargs='+', default=None,
                        help='List of low-rank ratios relative to block size or n_tokens (e.g., 1.0 0.75 0.5)')
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
    parser.add_argument('--lowrank_block_size', type=int, default=2,
                        help='Use blockwise low-rank projection with this block size (0 disables)')
    parser.add_argument('--gqa_num_q_heads', type=int, default=4)
    parser.add_argument('--gqa_num_kv_heads', type=int, default=2)
    parser.add_argument('--gla_disable_gate', action='store_true',
                        help='Disable GLA value gate (use raw V).')
    parser.add_argument('--gla_gate_bias', type=float, default=0.0,
                        help='Bias for GLA gate (sigmoid). Positive keeps gate open.')
    parser.add_argument('--gla_hidden_mult', type=float, default=None,
                        help='If set and hidden_dim is None, use hidden_dim = mult * d for GLA.')
    parser.add_argument('--sparse_n_head', type=int, default=4)
    parser.add_argument('--sparse_window_size', type=int, default=None,
                        help='If None, set to cover all tokens for the current n_points.')
    parser.add_argument('--sparse_stride', type=int, default=0)
    parser.add_argument('--sparse_global_tokens', type=int, default=None,
                        help='If None, set to cover all tokens for the current n_points.')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if args.compile and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (7, 0):
            print(f"[warn] torch.compile disabled: GPU capability {major}.{minor} < 7.0")
            args.compile = False
        else:
            try:
                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
            except Exception:
                pass
    
    print(f"Using device: {device}")
    
    # Calculate n_points = 2d + 1
    n_points = 2 * args.d + 1
    n_tokens = n_points + 1
    block_size = args.lowrank_block_size if args.lowrank_block_size and args.lowrank_block_size > 0 else None
    if args.lowrank_k_ratios and len(args.lowrank_k_ratios) > 0:
        lowrank_ratios = args.lowrank_k_ratios
    else:
        lowrank_ratios = [args.lowrank_k_ratio]
    
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

        def train_and_store(model, name):
            print(f"\nModel ({name}) parameters: {sum(p.numel() for p in model.parameters()):,}")
            if args.compile:
                try:
                    model = torch.compile(model)
                    print("Compiled model with torch.compile")
                except Exception as e:
                    print(f"[warn] torch.compile failed: {e}")
            trained_model, train_losses = train_model(
                model=model,
                d=args.d,
                n_points=n_points,
                num_tasks=args.num_train_tasks,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                sigma=args.sigma,
                device=device,
                use_amp=args.use_amp
            )
            models[name] = trained_model
            train_losses_by_model[name] = train_losses

            checkpoint_path = output_dir / f'{name}_{num_layers}layer_checkpoint.pt'
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'num_layers': num_layers,
                'd': args.d,
                'hidden_dim': args.hidden_dim,
                'train_losses': train_losses,
                'model_type': name
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        for model_type in args.model_types:
            if model_type == 'lsa':
                model = MultiLayerLSA(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim
                )
                train_and_store(model, 'lsa')
            elif model_type == 'gla':
                gla_hidden = args.hidden_dim
                if gla_hidden is None and args.gla_hidden_mult is not None:
                    gla_hidden = int(args.gla_hidden_mult * args.d)
                model = MultiLayerGLA(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=gla_hidden,
                    disable_gate=args.gla_disable_gate,
                    gate_bias=args.gla_gate_bias,
                )
                train_and_store(model, 'gla')
            elif model_type == 'gqa':
                model = GQATransformer(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim if args.hidden_dim is not None else args.d,
                    num_q_heads=args.gqa_num_q_heads,
                    num_kv_heads=args.gqa_num_kv_heads,
                )
                train_and_store(model, 'gqa')
            elif model_type == 'sparse':
                sparse_window = args.sparse_window_size
                if sparse_window is None:
                    sparse_window = 2 * n_points + 2
                sparse_global = args.sparse_global_tokens
                if sparse_global is None:
                    sparse_global = 2 * n_points + 2
                model = SparseICLModel(
                    d=args.d,
                    n_points=n_points,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim if args.hidden_dim is not None else args.d,
                    n_head=args.sparse_n_head,
                    window_size=sparse_window,
                    stride=args.sparse_stride,
                    global_tokens=sparse_global,
                )
                train_and_store(model, 'sparse')
            elif model_type == 'softmax':
                model = MultiLayerAttentionModel(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim,
                    attn_type='softmax',
                    causal=True
                )
                train_and_store(model, 'softmax')
            elif model_type in ('linformer', 'lowrank'):
                for ratio in lowrank_ratios:
                    if block_size is None:
                        proj_k = max(1, int(round(n_tokens * ratio)))
                        name = f"lowrank_k{ratio:g}"
                    else:
                        proj_k = max(1, int(round(block_size * ratio)))
                        name = f"lowrank_block{block_size}_k{ratio:g}"

                    identity_proj = args.lowrank_identity_proj and abs(ratio - 1.0) < 1e-9
                    if args.lowrank_exact_match and abs(ratio - 1.0) < 1e-9:
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
                            'block_size': block_size
                        }
                    )
                    train_and_store(model, name)
            elif model_type == 'kernel':
                model = MultiLayerAttentionModel(
                    d=args.d,
                    num_layers=num_layers,
                    hidden_dim=args.hidden_dim,
                    attn_type='kernel',
                    causal=True
                )
                train_and_store(model, 'kernel')
            else:
                raise ValueError(f"Unknown model type: {model_type}")

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
    lowrank_keys = set()
    for res in all_results:
        for k in res.keys():
            if k.startswith("mse_lowrank_") and k.endswith("_mean"):
                lowrank_keys.add(k.replace("_mean", ""))
    lowrank_keys = sorted(lowrank_keys)

    header = f"{'Layers':<10} {'MSE GD':<20} {'MSE LSA':<20} {'MSE Softmax':<20} {'MSE Kernel':<20}"
    for lk in lowrank_keys:
        header += f" {lk.replace('mse_', ''):<20}"
    print(header)
    print("-"*70)
    for res in all_results:
        row = (f"{res['num_layers']:<10} "
               f"{res['mse_gd_mean']:.6f} ± {res['mse_gd_std']:.4f}    "
               f"{res.get('mse_lsa_mean', float('nan')):.6f} ± {res.get('mse_lsa_std', float('nan')):.4f}    "
               f"{res.get('mse_softmax_mean', float('nan')):.6f} ± {res.get('mse_softmax_std', float('nan')):.4f}    "
               f"{res.get('mse_kernel_mean', float('nan')):.6f} ± {res.get('mse_kernel_std', float('nan')):.4f}")
        for lk in lowrank_keys:
            mean = res.get(f"{lk}_mean", float('nan'))
            std = res.get(f"{lk}_std", float('nan'))
            row += f" {mean:.6f} ± {std:.4f}"
        print(row)
    print("="*70)


if __name__ == '__main__':
    main()
