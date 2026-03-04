#!/usr/bin/env python3
"""
Layer sweep with per-layer training steps and incremental all_results.json updates.

This script trains each attention mechanism for each layer count, evaluates it,
and updates all_results.json after EACH model finishes. This allows live progress.
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


def train_model_steps(model, steps, d, n_points, batch_size, lr, sigma, device, use_amp=False, log_every=100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    losses = []

    pbar = tqdm(range(steps), desc="Training steps")
    for step in pbar:
        xs, ys, w_true = generate_linear_regression_task(
            d, n_points, batch_size=batch_size, sigma=sigma, device=device
        )
        query_x = torch.randn(batch_size, 1, d, device=device)
        query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        if sigma > 0:
            query_y = query_y + torch.randn_like(query_y) * sigma

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            y_pred = model(xs, ys, query_x)
            loss = criterion(y_pred, query_y)
        scaler.scale(loss).backward()
        if use_amp and device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % log_every == 0 or step == steps - 1:
            losses.append(float(loss.item()))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return model, losses


def evaluate_single_model(model, d, n_points, num_eval_tasks=1000,
                          batch_size=100, eta=0.1, sigma=0.0, device='cpu'):
    model.to(device)
    model.eval()

    mse_list = []
    cos_list = []
    mse_gd_list = []

    num_batches = num_eval_tasks // batch_size
    num_layers = model.num_layers

    with torch.no_grad():
        for _ in range(num_batches):
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

            y_pred = model(xs, ys, query_x)
            mse = ((y_pred - query_y) ** 2).mean().item()
            mse_list.append(mse)

            w_model = model.get_weight_update(xs, ys, query_x)
            cos = compute_cosine_similarity(w_model, w_gd).mean().item()
            cos_list.append(cos)

    return {
        "mse_gd_mean": float(np.mean(mse_gd_list)),
        "mse_gd_std": float(np.std(mse_gd_list)),
        "mse_mean": float(np.mean(mse_list)),
        "mse_std": float(np.std(mse_list)),
        "cosine_mean": float(np.mean(cos_list)),
        "cosine_std": float(np.std(cos_list)),
    }


def _load_all_results(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _update_all_results(path: Path, layer, model_name, metrics, train_losses):
    data = _load_all_results(path)
    by_layer = {r["num_layers"]: r for r in data if "num_layers" in r}

    if layer not in by_layer:
        by_layer[layer] = {"num_layers": layer}

    entry = by_layer[layer]
    # keep gd baseline if already there, else set
    if "mse_gd_mean" not in entry:
        entry["mse_gd_mean"] = metrics["mse_gd_mean"]
        entry["mse_gd_std"] = metrics["mse_gd_std"]

    entry[f"mse_{model_name}_mean"] = metrics["mse_mean"]
    entry[f"mse_{model_name}_std"] = metrics["mse_std"]
    entry[f"cosine_sim_{model_name}_mean"] = metrics["cosine_mean"]
    entry[f"cosine_sim_{model_name}_std"] = metrics["cosine_std"]
    entry[f"train_losses_{model_name}"] = train_losses

    out = [by_layer[k] for k in sorted(by_layer.keys())]
    path.write_text(json.dumps(out, indent=2))


def build_model(args, model_type, num_layers, n_points):
    n_tokens = n_points + 1
    block_size = args.lowrank_block_size if args.lowrank_block_size and args.lowrank_block_size > 0 else None
    if block_size is not None and n_tokens % block_size != 0:
        block_size = None

    if model_type == "lsa":
        return MultiLayerLSA(d=args.d, num_layers=num_layers, hidden_dim=args.hidden_dim)
    if model_type == "gla":
        gla_hidden = args.hidden_dim
        if gla_hidden is None and args.gla_hidden_mult is not None:
            gla_hidden = int(args.gla_hidden_mult * args.d)
        return MultiLayerGLA(
            d=args.d, num_layers=num_layers, hidden_dim=gla_hidden,
            disable_gate=args.gla_disable_gate, gate_bias=args.gla_gate_bias
        )
    if model_type == "gqa":
        return GQATransformer(
            d=args.d, num_layers=num_layers,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else args.d,
            num_q_heads=args.gqa_num_q_heads, num_kv_heads=args.gqa_num_kv_heads
        )
    if model_type == "sparse":
        sparse_window = args.sparse_window_size
        if sparse_window is None:
            sparse_window = 2 * n_points + 2
        sparse_global = args.sparse_global_tokens
        if sparse_global is None:
            sparse_global = 2 * n_points + 2
        return SparseICLModel(
            d=args.d, n_points=n_points, num_layers=num_layers,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else args.d,
            n_head=args.sparse_n_head, window_size=sparse_window,
            stride=args.sparse_stride, global_tokens=sparse_global
        )
    if model_type == "softmax":
        return MultiLayerAttentionModel(
            d=args.d, num_layers=num_layers, hidden_dim=args.hidden_dim,
            attn_type="softmax", causal=True
        )
    if model_type == "kernel":
        return MultiLayerAttentionModel(
            d=args.d, num_layers=num_layers, hidden_dim=args.hidden_dim,
            attn_type="kernel", causal=True
        )
    if model_type == "linformer":
        models = []
        lowrank_ratios = args.lowrank_k_ratios or [args.lowrank_k_ratio]
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
            m = MultiLayerAttentionModel(
                d=args.d, num_layers=num_layers, hidden_dim=args.hidden_dim,
                attn_type="linformer", n_tokens=n_tokens, proj_k=proj_k,
                lowrank_kwargs={
                    "share_ef": args.lowrank_share_ef,
                    "orth_init": args.lowrank_orth_init,
                    "identity_proj": identity_proj,
                    "freeze_proj": freeze_proj,
                    "normalize_qk": normalize_qk,
                    "learnable_scale": learnable_scale,
                    "block_size": block_size,
                }
            )
            models.append((name, m))
        return models
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Layer sweep with per-layer step schedule")
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--num_layers_list", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--n_points", type=int, default=41)
    parser.add_argument("--steps_shallow", type=int, default=5000)
    parser.add_argument("--steps_deep", type=int, default=10000)
    parser.add_argument("--deep_layer_threshold", type=int, default=16,
                        help="Layers >= this use steps_deep")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--num_eval_tasks", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--model_types", type=str, nargs="+",
                        default=["lsa", "softmax", "linformer", "kernel", "gla", "gqa", "sparse"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--lowrank_k_ratio", type=float, default=0.5)
    parser.add_argument("--lowrank_k_ratios", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--lowrank_share_ef", action="store_true")
    parser.add_argument("--lowrank_orth_init", action="store_true")
    parser.add_argument("--lowrank_identity_proj", action="store_true")
    parser.add_argument("--lowrank_freeze_proj", action="store_true")
    parser.add_argument("--lowrank_exact_match", action="store_true")
    parser.add_argument("--lowrank_block_size", type=int, default=2)
    parser.add_argument("--gqa_num_q_heads", type=int, default=4)
    parser.add_argument("--gqa_num_kv_heads", type=int, default=2)
    parser.add_argument("--gla_disable_gate", action="store_true")
    parser.add_argument("--gla_gate_bias", type=float, default=0.0)
    parser.add_argument("--gla_hidden_mult", type=float, default=None)
    parser.add_argument("--sparse_n_head", type=int, default=4)
    parser.add_argument("--sparse_window_size", type=int, default=None)
    parser.add_argument("--sparse_stride", type=int, default=0)
    parser.add_argument("--sparse_global_tokens", type=int, default=None)

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "all_results.json"

    for num_layers in args.num_layers_list:
        steps = args.steps_deep if num_layers >= args.deep_layer_threshold else args.steps_shallow
        print(f"\n{'='*60}")
        print(f"Layers = {num_layers} | steps = {steps}")
        print(f"{'='*60}")

        # build list of (model_name, model)
        models_to_run = []
        for mt in args.model_types:
            if mt == "linformer":
                models_to_run.extend(build_model(args, mt, num_layers, args.n_points))
            else:
                models_to_run.append((mt, build_model(args, mt, num_layers, args.n_points)))

        for model_name, model in models_to_run:
            print(f"\nTraining model: {model_name}")
            model = model.to(device)
            trained_model, losses = train_model_steps(
                model, steps, d=args.d, n_points=args.n_points,
                batch_size=args.batch_size, lr=args.lr, sigma=args.sigma,
                device=device, use_amp=args.use_amp
            )

            ckpt_path = output_dir / f"{model_name}_{num_layers}layer_checkpoint.pt"
            torch.save({
                "model_state_dict": trained_model.state_dict(),
                "num_layers": num_layers,
                "d": args.d,
                "hidden_dim": args.hidden_dim,
                "train_losses": losses,
                "model_type": model_name,
            }, ckpt_path)

            metrics = evaluate_single_model(
                trained_model, d=args.d, n_points=args.n_points,
                num_eval_tasks=args.num_eval_tasks, batch_size=100,
                eta=args.eta, sigma=args.sigma, device=device
            )
            _update_all_results(results_path, num_layers, model_name, metrics, losses)
            print(f"Updated {results_path} after {model_name} @ {num_layers} layers")

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
