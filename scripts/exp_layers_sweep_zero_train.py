#!/usr/bin/env python3
"""
Experiment: layer sweep with ZERO training (random init).

Outputs:
  - all_results.json (list of dicts keyed by num_layers)
  - results_layers_<L>.json per layer count
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.lsa import MultiLayerLSA
from src.models.attention_variants import MultiLayerAttentionModel
from src.evaluation.gd_baseline import (
    generate_linear_regression_task,
    gd_t_steps,
    compute_cosine_similarity
)


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


def build_models(args, n_points, num_layers, device):
    n_tokens = n_points + 1
    block_size = args.lowrank_block_size if args.lowrank_block_size and args.lowrank_block_size > 0 else None
    if block_size is not None and n_tokens % block_size != 0:
        print(f"[warn] block_size={block_size} does not divide n_tokens={n_tokens}; using global projection.")
        block_size = None

    if args.lowrank_k_ratios and len(args.lowrank_k_ratios) > 0:
        lowrank_ratios = args.lowrank_k_ratios
    else:
        lowrank_ratios = [args.lowrank_k_ratio]

    models = {}
    if 'lsa' in args.model_types:
        models['lsa'] = MultiLayerLSA(d=args.d, num_layers=num_layers, hidden_dim=args.hidden_dim)
    if 'softmax' in args.model_types:
        models['softmax'] = MultiLayerAttentionModel(
            d=args.d,
            num_layers=num_layers,
            hidden_dim=args.hidden_dim,
            attn_type='softmax',
            causal=True
        )
    if 'kernel' in args.model_types:
        models['kernel'] = MultiLayerAttentionModel(
            d=args.d,
            num_layers=num_layers,
            hidden_dim=args.hidden_dim,
            attn_type='kernel',
            causal=True
        )
    if 'linformer' in args.model_types or 'lowrank' in args.model_types:
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

            models[name] = MultiLayerAttentionModel(
                d=args.d,
                num_layers=num_layers,
                hidden_dim=args.hidden_dim,
                attn_type='linformer',
                n_tokens=n_tokens,
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

    for m in models.values():
        m.to(device)
    return models


def main():
    parser = argparse.ArgumentParser(description="Layer sweep with zero training")
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--num_layers_list', type=int, nargs='+',
                        default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_eval_tasks', type=int, default=1000)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str,
                        default='./results/exp_all/exp_layers_sweep_zero')
    parser.add_argument('--device', type=str, default='auto')

    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['lsa', 'softmax', 'linformer', 'kernel'])
    parser.add_argument('--lowrank_k_ratio', type=float, default=0.5,
                        help='Single low-rank ratio (backward compatible). Ignored if --lowrank_k_ratios is set.')
    parser.add_argument('--lowrank_k_ratios', type=float, nargs='+', default=[1.0, 0.75, 0.5])
    parser.add_argument('--lowrank_share_ef', action='store_true')
    parser.add_argument('--lowrank_orth_init', action='store_true')
    parser.add_argument('--lowrank_identity_proj', action='store_true')
    parser.add_argument('--lowrank_freeze_proj', action='store_true')
    parser.add_argument('--lowrank_exact_match', action='store_true')
    parser.add_argument('--lowrank_block_size', type=int, default=2)

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    n_points = 2 * args.d + 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for num_layers in args.num_layers_list:
        print(f"\n{'='*60}")
        print(f"Experiment: num_layers = {num_layers} (zero training)")
        print(f"{'='*60}")

        models = build_models(args, n_points, num_layers, device)

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
        results['train_steps'] = 0
        all_results.append(results)

        with open(output_dir / f"results_layers_{num_layers}.json", 'w') as f:
            json.dump(results, f, indent=2)

    all_results = sorted(all_results, key=lambda r: r.get('num_layers', 0))
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {output_dir}")


if __name__ == '__main__':
    main()
