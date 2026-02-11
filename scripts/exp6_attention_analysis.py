#!/usr/bin/env python3
"""Experiment 6: Attention pattern analysis."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.task_sampler import LinearRegressionTaskSampler
from src.evaluation.gd_alignment import compute_gd_alignment
from src.experiments.runner import load_model_from_checkpoint, save_experiment_result, train_and_get_checkpoint


def get_attention_matrix(model, batch):
    model.eval()
    with torch.no_grad():
        _, attn_list = model.forward(
            batch["x_context"],
            batch["y_context"],
            batch["x_query"],
            return_attn=True,
        )
    if not attn_list:
        raise RuntimeError("No attention map returned")
    # Last layer, first head.
    return attn_list[-1][0, 0].detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp6 attention analysis")
    parser.add_argument("--config", type=str, default="configs/train_baseline_d20.json")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--output", type=str, default="results/exp6_attention_analysis")
    args = parser.parse_args()

    base = load_config(args.config)
    base["train"]["total_steps"] = args.steps
    base["model"]["task_dim"] = 20
    base["model"]["max_context"] = 20
    base["eval"]["compute_baselines"] = False

    mechanisms = {
        "baseline": {"attn_type": "baseline"},
        "linear": {"attn_type": "linear"},
        "lowrank_r16": {"attn_type": "lowrank", "attn_rank": 16},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = LinearRegressionTaskSampler(task_dim=20, device=device)
    fixed_batch = sampler.sample_batch(
        batch_size=1,
        n_context=20,
        active_dim=20,
        noise_std=0.0,
        x_distribution="isotropic",
        skew_power=2.0,
        orthant_mode="none",
    )

    results = {}

    fig, axes = plt.subplots(2, len(mechanisms), figsize=(5 * len(mechanisms), 8))

    for col, (name, mech_cfg) in enumerate(mechanisms.items()):
        cfg = deepcopy(base)
        cfg["model"].update(mech_cfg)
        cfg["output_dir"] = str(Path(args.output) / name)

        print(f"\n[Exp6] Training {name}")
        ckpt = train_and_get_checkpoint(cfg)
        model = load_model_from_checkpoint(cfg, ckpt).to(device)

        attn = get_attention_matrix(model, fixed_batch)
        svals = torch.linalg.svdvals(attn)

        align = compute_gd_alignment(
            model=model,
            x_context=fixed_batch["x_context"],
            y_context=fixed_batch["y_context"],
            task_dim=20,
            gd_eta=1.0,
        )
        cosine = float(align["cosine"].mean().item())

        results[name] = {
            "checkpoint": str(ckpt),
            "cosine_alignment": cosine,
            "singular_values": svals.tolist(),
        }

        ax_hm = axes[0, col]
        hm = ax_hm.imshow(attn.numpy(), aspect="auto", cmap="viridis")
        ax_hm.set_title(f"{name} attention")
        ax_hm.set_xlabel("key index")
        ax_hm.set_ylabel("query index")
        fig.colorbar(hm, ax=ax_hm, fraction=0.046, pad=0.04)

        ax_sv = axes[1, col]
        ax_sv.plot(svals.numpy(), marker="o", markersize=2)
        ax_sv.set_title(f"{name} singular values")
        ax_sv.set_xlabel("index")
        ax_sv.set_ylabel("sigma")
        ax_sv.grid(alpha=0.3)

        print(f"  cosine alignment={cosine:.4f}")

    save_experiment_result(args.output, "results.json", results)

    out_path = Path(args.output) / "exp6_attention_analysis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
