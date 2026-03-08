#!/usr/bin/env python3
"""
Experiment: sweep training steps (optimizer updates) and evaluate GD alignment.

Outputs:
  - all_results.json (list of dicts keyed by train_steps)
  - results_steps_<N>.json per checkpoint
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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
    if 'gla' in args.model_types:
        gla_hidden = args.hidden_dim
        if gla_hidden is None and args.gla_hidden_mult is not None:
            gla_hidden = int(args.gla_hidden_mult * args.d)
        models['gla'] = MultiLayerGLA(
            d=args.d,
            num_layers=num_layers,
            hidden_dim=gla_hidden,
            disable_gate=args.gla_disable_gate,
            gate_bias=args.gla_gate_bias,
        )
    if 'gqa' in args.model_types:
        models['gqa'] = GQATransformer(
            d=args.d,
            num_layers=num_layers,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else args.d,
            num_q_heads=args.gqa_num_q_heads,
            num_kv_heads=args.gqa_num_kv_heads,
        )
    if 'sparse' in args.model_types:
        sparse_window = args.sparse_window_size
        if sparse_window is None:
            sparse_window = 2 * n_points + 2
        sparse_global = args.sparse_global_tokens
        if sparse_global is None:
            sparse_global = 2 * n_points + 2
        models['sparse'] = SparseICLModel(
            d=args.d,
            n_points=n_points,
            num_layers=num_layers,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else args.d,
            n_head=args.sparse_n_head,
            window_size=sparse_window,
            stride=args.sparse_stride,
            global_tokens=sparse_global,
        )
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


def run_steps_sweep(models, steps_list, d, n_points, batch_size, lr, sigma,
                    num_eval_tasks, eta, device, use_amp=False):
    steps_list = sorted(set(int(s) for s in steps_list))
    max_steps = max(steps_list)
    steps_set = set(steps_list)

    optimizers = {name: torch.optim.Adam(m.parameters(), lr=lr) for name, m in models.items()}
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    all_results = []

    # Evaluate at step 0 (random init) if requested
    if 0 in steps_set:
        print("\nEvaluating at step 0")
        results = evaluate_models(
            models=models,
            d=d,
            n_points=n_points,
            num_eval_tasks=num_eval_tasks,
            batch_size=100,
            eta=eta,
            sigma=sigma,
            device=device
        )
        results['train_steps'] = 0
        all_results.append(results)

    pbar = tqdm(total=max_steps, desc="Training steps")
    current_step = 0
    while current_step < max_steps:
        xs, ys, w_true = generate_linear_regression_task(
            d, n_points, batch_size=batch_size, sigma=sigma, device=device
        )
        query_x = torch.randn(batch_size, 1, d, device=device)
        query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        if sigma > 0:
            query_y = query_y + torch.randn_like(query_y) * sigma

        for name, model in models.items():
            optimizers[name].zero_grad()
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                y_pred = model(xs, ys, query_x)
                loss = criterion(y_pred, query_y)
            scaler.scale(loss).backward()
            if use_amp and device.type == "cuda":
                scaler.unscale_(optimizers[name])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizers[name])
            scaler.update()

        current_step += 1
        pbar.update(1)

        if current_step in steps_set:
            print(f"\nEvaluating at step {current_step}")
            results = evaluate_models(
                models=models,
                d=d,
                n_points=n_points,
                num_eval_tasks=num_eval_tasks,
                batch_size=100,
                eta=eta,
                sigma=sigma,
                device=device
            )
            results['train_steps'] = current_step
            all_results.append(results)

    pbar.close()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Training-steps sweep for attention mechanisms")
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n_points', type=int, default=41,
                        help='Number of in-context examples (n)')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--train_steps_list', type=int, nargs='+',
                        default=[0, 1000, 2000, 5000, 10000, 20000])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--num_eval_tasks', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='./results/exp_steps_sweep')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision (AMP) on CUDA')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile on models (PyTorch 2.x)')

    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['lsa', 'softmax', 'linformer', 'kernel', 'gla', 'gqa', 'sparse'],
                        help='Model types to run: lsa, softmax, linformer, kernel')
    parser.add_argument('--lowrank_k_ratio', type=float, default=0.5,
                        help='Single low-rank ratio (backward compatible). Ignored if --lowrank_k_ratios is set.')
    parser.add_argument('--lowrank_k_ratios', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='List of low-rank ratios relative to block size or n_tokens')
    parser.add_argument('--lowrank_share_ef', action='store_true')
    parser.add_argument('--lowrank_orth_init', action='store_true')
    parser.add_argument('--lowrank_identity_proj', action='store_true')
    parser.add_argument('--lowrank_freeze_proj', action='store_true')
    parser.add_argument('--lowrank_exact_match', action='store_true')
    parser.add_argument('--lowrank_block_size', type=int, default=2)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(args, args.n_points, args.num_layers, device)
    if args.compile:
        try:
            models = {k: torch.compile(v) for k, v in models.items()}
            print("Compiled models with torch.compile")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    # Train to max steps, evaluate at checkpoints
    all_results = run_steps_sweep(
        models=models,
        steps_list=args.train_steps_list,
        d=args.d,
        n_points=args.n_points,
        batch_size=args.batch_size,
        lr=args.lr,
        sigma=args.sigma,
        num_eval_tasks=args.num_eval_tasks,
        eta=args.eta,
        device=device,
        use_amp=args.use_amp
    )

    all_results = sorted(all_results, key=lambda r: r.get('train_steps', 0))
    for res in all_results:
        step = res['train_steps']
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"results_steps_{step}.json", 'w') as f:
            json.dump(res, f, indent=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {output_dir}")


if __name__ == '__main__':
    main()
