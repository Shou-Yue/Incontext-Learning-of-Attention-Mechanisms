#!/usr/bin/env python3
"""
Compare Softmax vs Low-rank Softmax vs GD with configurable low-rank settings.

Supports:
  - proj_k as fraction of n_tokens (e.g., 1.0, 0.75, 0.5)
  - E/F sharing
  - orthogonal initialization for E/F
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.attention_variants import MultiLayerAttentionModel
from src.models.lsa import MultiLayerLSA
from src.evaluation.gd_baseline import (
    generate_linear_regression_task,
    gd_t_steps,
    compute_cosine_similarity
)


def train_model(model, d, n_points, num_tasks=10000, batch_size=64,
                num_epochs=10, lr=1e-3, sigma=0.0, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_batches = num_tasks // batch_size
    for epoch in range(num_epochs):
        epoch_losses = []
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for _ in pbar:
            xs, ys, w_true = generate_linear_regression_task(
                d, n_points, batch_size=batch_size, sigma=sigma, device=device
            )
            query_x = torch.randn(batch_size, 1, d, device=device)
            query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            if sigma > 0:
                query_y = query_y + torch.randn_like(query_y) * sigma

            optimizer.zero_grad()
            y_pred = model(xs, ys, query_x)
            loss = criterion(y_pred, query_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        print(f"Epoch {epoch+1} avg loss: {np.mean(epoch_losses):.6f}")

    return model


def evaluate_models(models, d, n_points, num_eval_tasks=1000,
                    batch_size=100, eta=0.1, sigma=0.0, device='cpu'):
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

            # GD baseline
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
    parser = argparse.ArgumentParser(description="Low-rank vs Softmax vs GD sweep")
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--num_layers_list', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_train_tasks', type=int, default=10000)
    parser.add_argument('--num_eval_tasks', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default='./results/lowrank_sweep')
    parser.add_argument('--device', type=str, default='auto')

    parser.add_argument('--proj_k_ratios', type=float, nargs='+',
                        default=[1.0, 0.75, 0.5],
                        help='Fractions of tokens per block for low-rank k')
    parser.add_argument('--block_size', type=int, default=2,
                        help='Block size for local projections (use 0 to disable)')
    parser.add_argument('--include_kernel', action='store_true',
                        help='Include kernelized linear attention baseline')
    parser.add_argument('--include_lsa', action='store_true',
                        help='Include LSA baseline (linear self-attention)')
    parser.add_argument('--share_ef', action='store_true',
                        help='Share E and F projection matrices')
    parser.add_argument('--orth_init', action='store_true',
                        help='Use orthogonal initialization for E/F')
    parser.add_argument('--identity_proj', action='store_true',
                        help='For ratio=1.0, set E/F to identity (requires proj_k == n_tokens)')
    parser.add_argument('--freeze_proj', action='store_true',
                        help='Freeze E/F projection matrices')
    parser.add_argument('--exact_match', action='store_true',
                        help='Force low-rank k=1.0 to match softmax (identity, no norm, fixed scale)')

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    n_points = 2 * args.d + 1
    n_tokens = n_points + 1
    block_size = None if args.block_size == 0 else args.block_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for num_layers in args.num_layers_list:
        print(f"\n{'='*60}")
        print(f"Experiment: num_layers = {num_layers}")
        print(f"{'='*60}")

        models = {}

        # Softmax baseline
        softmax_model = MultiLayerAttentionModel(
            d=args.d,
            num_layers=num_layers,
            hidden_dim=args.hidden_dim,
            attn_type='softmax',
            causal=False
        )
        print(f"\nModel (softmax) parameters: {sum(p.numel() for p in softmax_model.parameters()):,}")
        softmax_model = train_model(
            model=softmax_model,
            d=args.d,
            n_points=n_points,
            num_tasks=args.num_train_tasks,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            sigma=args.sigma,
            device=device
        )
        models['softmax'] = softmax_model

        # Kernelized linear attention baseline
        if args.include_kernel:
            kernel_model = MultiLayerAttentionModel(
                d=args.d,
                num_layers=num_layers,
                hidden_dim=args.hidden_dim,
                attn_type='kernel',
                causal=False
            )
            print(f"\nModel (kernel) parameters: {sum(p.numel() for p in kernel_model.parameters()):,}")
            kernel_model = train_model(
                model=kernel_model,
                d=args.d,
                n_points=n_points,
                num_tasks=args.num_train_tasks,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                sigma=args.sigma,
                device=device
            )
            models['kernel'] = kernel_model

        # LSA baseline
        if args.include_lsa:
            lsa_model = MultiLayerLSA(
                d=args.d,
                num_layers=num_layers,
                hidden_dim=args.hidden_dim
            )
            print(f"\nModel (lsa) parameters: {sum(p.numel() for p in lsa_model.parameters()):,}")
            lsa_model = train_model(
                model=lsa_model,
                d=args.d,
                n_points=n_points,
                num_tasks=args.num_train_tasks,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                sigma=args.sigma,
                device=device
            )
            models['lsa'] = lsa_model

        # Low-rank variants
        for ratio in args.proj_k_ratios:
            if block_size is None:
                proj_k = max(1, int(round(n_tokens * ratio)))
                name = f"lowrank_k{ratio:g}"
            else:
                proj_k = max(1, int(round(block_size * ratio)))
                name = f"lowrank_block{block_size}_k{ratio:g}"
            identity_proj = args.identity_proj and abs(ratio - 1.0) < 1e-9
            if args.exact_match and abs(ratio - 1.0) < 1e-9:
                identity_proj = True
                freeze_proj = True
                normalize_qk = False
                learnable_scale = False
            else:
                freeze_proj = args.freeze_proj
                normalize_qk = True
                learnable_scale = True
            lowrank_model = MultiLayerAttentionModel(
                d=args.d,
                num_layers=num_layers,
                hidden_dim=args.hidden_dim,
                attn_type='linformer',
                n_tokens=n_tokens,
                proj_k=proj_k,
                lowrank_kwargs={
                    'share_ef': args.share_ef,
                    'orth_init': args.orth_init,
                    'identity_proj': identity_proj,
                    'freeze_proj': freeze_proj,
                    'normalize_qk': normalize_qk,
                    'learnable_scale': learnable_scale,
                    'block_size': block_size
                }
            )
            print(f"\nModel ({name}) parameters: {sum(p.numel() for p in lowrank_model.parameters()):,}")
            lowrank_model = train_model(
                model=lowrank_model,
                d=args.d,
                n_points=n_points,
                num_tasks=args.num_train_tasks,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                sigma=args.sigma,
                device=device
            )
            models[name] = lowrank_model

        results = evaluate_models(
            models=models,
            d=args.d,
            n_points=n_points,
            num_eval_tasks=args.num_eval_tasks,
            batch_size=max(10, args.batch_size // 2),
            eta=args.eta,
            sigma=args.sigma,
            device=device
        )

        results['num_layers'] = num_layers
        results['proj_k_ratios'] = args.proj_k_ratios
        results['share_ef'] = args.share_ef
        results['orth_init'] = args.orth_init
        all_results.append(results)

        with open(output_dir / f"results_layers_{num_layers}.json", 'w') as f:
            json.dump(results, f, indent=2)

    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
