#!/usr/bin/env python3
"""
Rebuild exp_layers_sweep/all_results.json from existing checkpoints.

This is useful when training finished but evaluation/all_results.json
is incomplete. It loads checkpoints, evaluates models, and writes a
fresh all_results.json.
"""
import argparse
import json
import re
from pathlib import Path
import sys

import torch

# add repo root to path for src imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lsa import MultiLayerLSA
from src.models.attention_variants import MultiLayerAttentionModel
from src.models.gla import MultiLayerGLA
from src.models.gqa import GQATransformer
from src.models.sparse_causal import SparseICLModel
from scripts.lsa_gd_multilayer import evaluate_models


def _parse_ckpt_name(filename: str):
    if not filename.endswith("_checkpoint.pt"):
        return None
    base = filename[:-len("_checkpoint.pt")]
    parts = base.split("_")
    if not parts:
        return None
    layer_token = parts[-1]  # e.g., "8layer"
    if not layer_token.endswith("layer"):
        return None
    try:
        num_layers = int(layer_token.replace("layer", ""))
    except ValueError:
        return None
    model_name = "_".join(parts[:-1])
    return model_name, num_layers


def _parse_lowrank(model_name: str):
    # lowrank_block2_k0.5 or lowrank_k0.5
    m = re.match(r"lowrank_block(\d+)_k([0-9.]+)", model_name)
    if m:
        return int(m.group(1)), float(m.group(2))
    m = re.match(r"lowrank_k([0-9.]+)", model_name)
    if m:
        return None, float(m.group(1))
    return None, None


def _build_model(model_name, num_layers, d, n_tokens, hidden_dim, gqa_q, gqa_kv,
                 sparse_n_head, sparse_window, sparse_stride, sparse_global,
                 gla_disable_gate, gla_gate_bias, gla_hidden_mult, device):
    if model_name == "lsa":
        model = MultiLayerLSA(d=d, num_layers=num_layers, hidden_dim=hidden_dim)
    elif model_name == "gla":
        gla_hidden = hidden_dim
        if gla_hidden is None and gla_hidden_mult is not None:
            gla_hidden = int(gla_hidden_mult * d)
        model = MultiLayerGLA(
            d=d,
            num_layers=num_layers,
            hidden_dim=gla_hidden,
            disable_gate=gla_disable_gate,
            gate_bias=gla_gate_bias,
        )
    elif model_name == "gqa":
        model = GQATransformer(
            d=d,
            num_layers=num_layers,
            hidden_dim=hidden_dim if hidden_dim is not None else d,
            num_q_heads=gqa_q,
            num_kv_heads=gqa_kv
        )
    elif model_name == "sparse":
        n_points = n_tokens - 1
        if sparse_window is None:
            sparse_window = 2 * n_tokens
        if sparse_global is None:
            sparse_global = 2 * n_tokens
        model = SparseICLModel(
            d=d,
            n_points=n_points,
            num_layers=num_layers,
            hidden_dim=hidden_dim if hidden_dim is not None else d,
            n_head=sparse_n_head,
            window_size=sparse_window,
            stride=sparse_stride,
            global_tokens=sparse_global,
        )
    elif model_name == "softmax":
        model = MultiLayerAttentionModel(
            d=d,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            attn_type="softmax",
            causal=True,
        )
    elif model_name == "kernel":
        model = MultiLayerAttentionModel(
            d=d,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            attn_type="kernel",
            causal=True,
        )
    elif model_name.startswith("lowrank_"):
        block_size, ratio = _parse_lowrank(model_name)
        if ratio is None:
            raise ValueError(f"Cannot parse lowrank model name: {model_name}")
        if block_size is None:
            proj_k = max(1, int(round(n_tokens * ratio)))
        else:
            proj_k = max(1, int(round(block_size * ratio)))
        model = MultiLayerAttentionModel(
            d=d,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            attn_type="linformer",
            n_tokens=n_tokens,
            proj_k=proj_k,
            lowrank_kwargs={
                "share_ef": False,
                "orth_init": False,
                "identity_proj": False,
                "freeze_proj": False,
                "normalize_qk": True,
                "learnable_scale": True,
                "block_size": block_size,
            }
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Rebuild layer sweep results from checkpoints")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to exp_layers_sweep directory")
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--num_eval_tasks", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--gqa_num_q_heads", type=int, default=4)
    parser.add_argument("--gqa_num_kv_heads", type=int, default=2)
    parser.add_argument("--gla_disable_gate", action="store_true",
                        help="Disable GLA value gate (use raw V).")
    parser.add_argument("--gla_gate_bias", type=float, default=0.0,
                        help="Bias for GLA gate (sigmoid). Positive keeps gate open.")
    parser.add_argument("--gla_hidden_mult", type=float, default=None,
                        help="If set and hidden_dim is None, use hidden_dim = mult * d for GLA.")
    parser.add_argument("--sparse_n_head", type=int, default=4)
    parser.add_argument("--sparse_window_size", type=int, default=None,
                        help="If None, set to cover all tokens for the current n_points.")
    parser.add_argument("--sparse_stride", type=int, default=0)
    parser.add_argument("--sparse_global_tokens", type=int, default=None,
                        help="If None, set to cover all tokens for the current n_points.")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    exp_dir = Path(args.exp_dir)
    ckpts = list(exp_dir.glob("*layer_checkpoint.pt"))
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {exp_dir}")

    # group checkpoints by layer
    groups = {}
    for ckpt in ckpts:
        parsed = _parse_ckpt_name(ckpt.name)
        if not parsed:
            continue
        model_name, num_layers = parsed
        groups.setdefault(num_layers, []).append((model_name, ckpt))

    n_points = 2 * args.d + 1
    n_tokens = n_points + 1

    all_results = []
    for num_layers in sorted(groups.keys()):
        models = {}
        train_losses_by_model = {}
        for model_name, ckpt_path in groups[num_layers]:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            hidden_dim = ckpt.get("hidden_dim", None)
            model = _build_model(
                model_name=model_name,
                num_layers=num_layers,
                d=args.d,
                n_tokens=n_tokens,
                hidden_dim=hidden_dim,
                gqa_q=args.gqa_num_q_heads,
                gqa_kv=args.gqa_num_kv_heads,
                sparse_n_head=args.sparse_n_head,
                sparse_window=args.sparse_window_size,
                sparse_stride=args.sparse_stride,
                sparse_global=args.sparse_global_tokens,
                gla_disable_gate=args.gla_disable_gate,
                gla_gate_bias=args.gla_gate_bias,
                gla_hidden_mult=args.gla_hidden_mult,
                device=device,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            models[model_name] = model
            if "train_losses" in ckpt:
                train_losses_by_model[model_name] = ckpt["train_losses"]

        results = evaluate_models(
            models=models,
            d=args.d,
            n_points=n_points,
            num_eval_tasks=args.num_eval_tasks,
            batch_size=args.batch_size,
            eta=args.eta,
            sigma=args.sigma,
            device=device,
        )
        results["num_layers"] = num_layers
        for model_name, losses in train_losses_by_model.items():
            results[f"train_losses_{model_name}"] = losses
        all_results.append(results)

    all_results = sorted(all_results, key=lambda r: r.get("num_layers", 0))
    out_path = exp_dir / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Rebuilt results at: {out_path}")


if __name__ == "__main__":
    main()
