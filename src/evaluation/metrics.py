"""Evaluation entrypoints for trained models."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch

from src.evaluation.baselines import run_all_baselines
from src.evaluation.gd_alignment import compute_gd_alignment


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    var = torch.var(target)
    return mse(pred, target) / (var + eps)


def evaluate_batches(
    model,
    sampler,
    num_prompts: int,
    batch_size: int,
    task_dim: int,
    n_context: int,
    active_dim: int,
    noise_std: float,
    x_distribution: str,
    skew_power: float,
    orthant_mode: str,
    gd_eta: float = 1.0,
    compute_baselines: bool = True,
    compute_alignment: bool = True,
    knn_k: int = 3,
    ridge_lambda: float = 1e-2,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    counts = defaultdict(int)

    if device is None:
        device = next(model.parameters()).device

    n_done = 0
    with torch.no_grad():
        while n_done < num_prompts:
            bsz = min(batch_size, num_prompts - n_done)
            batch = sampler.sample_batch(
                batch_size=bsz,
                n_context=n_context,
                active_dim=active_dim,
                noise_std=noise_std,
                x_distribution=x_distribution,
                skew_power=skew_power,
                orthant_mode=orthant_mode,
            )

            x_context = batch["x_context"].to(device)
            y_context = batch["y_context"].to(device)
            x_query = batch["x_query"].to(device)
            y_query = batch["y_query"].squeeze(1).to(device)

            y_hat = model.predict(x_context, y_context, x_query)
            batch_mse = mse(y_hat, y_query)
            batch_nmse = nmse(y_hat, y_query)

            totals["model_mse"] += batch_mse.item() * bsz
            totals["model_nmse"] += batch_nmse.item() * bsz
            counts["model"] += bsz

            if compute_baselines:
                baseline_preds = run_all_baselines(
                    x_context,
                    y_context,
                    x_query,
                    knn_k=knn_k,
                    ridge_lambda=ridge_lambda,
                    gd_eta=gd_eta,
                )
                for name, pred in baseline_preds.items():
                    value = mse(pred, y_query).item()
                    totals[f"{name}_mse"] += value * bsz
                    counts[name] += bsz

            if compute_alignment:
                align = compute_gd_alignment(
                    model=model,
                    x_context=x_context,
                    y_context=y_context,
                    task_dim=task_dim,
                    gd_eta=gd_eta,
                )
                totals["cosine_alignment"] += align["cosine"].mean().item() * bsz
                counts["cosine"] += bsz

            n_done += bsz

    out = {
        "mse": totals["model_mse"] / max(counts["model"], 1),
        "nmse": totals["model_nmse"] / max(counts["model"], 1),
    }
    if compute_baselines:
        for name in ["least_squares", "min_norm", "ridge", "knn", "averaging", "gd_one_step"]:
            out[f"{name}_mse"] = totals[f"{name}_mse"] / max(counts[name], 1)
    if compute_alignment:
        out["cosine_alignment"] = totals["cosine_alignment"] / max(counts["cosine"], 1)

    return out


def evaluate_grid(
    model,
    sampler,
    context_grid: Iterable[int],
    settings: Dict,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for n_context in context_grid:
        metrics = evaluate_batches(
            model=model,
            sampler=sampler,
            num_prompts=settings["num_prompts"],
            batch_size=settings["batch_size"],
            task_dim=settings["task_dim"],
            n_context=n_context,
            active_dim=settings.get("active_dim", settings["task_dim"]),
            noise_std=settings.get("noise_std", 0.0),
            x_distribution=settings.get("x_distribution", "isotropic"),
            skew_power=settings.get("skew_power", 2.0),
            orthant_mode=settings.get("orthant_mode", "none"),
            gd_eta=settings.get("gd_eta", 1.0),
            compute_baselines=settings.get("compute_baselines", True),
            compute_alignment=settings.get("compute_alignment", True),
            knn_k=settings.get("knn_k", 3),
            ridge_lambda=settings.get("ridge_lambda", 1e-2),
        )
        results[str(n_context)] = metrics
    return results
