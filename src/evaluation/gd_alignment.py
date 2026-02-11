"""Gradient-descent alignment utilities for ICL models."""
from __future__ import annotations

from typing import Dict

import torch


def gd_one_step_weights(x_context: torch.Tensor, y_context: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    """One-step GD from w0=0 under MSE loss: w1 = (eta/n) X^T y."""
    n_context = x_context.size(1)
    return eta * torch.einsum("bn,bnd->bd", y_context, x_context) / n_context


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)


def extract_implicit_weights(
    model,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    task_dim: int,
) -> torch.Tensor:
    """Estimate implicit linear weights by querying model on basis vectors."""
    device = x_context.device
    batch_size = x_context.size(0)
    basis = torch.eye(task_dim, device=device)

    preds = torch.zeros(batch_size, task_dim, device=device)
    with torch.no_grad():
        for i in range(task_dim):
            x_query = basis[i].view(1, 1, task_dim).expand(batch_size, 1, task_dim)
            y_pred = model.predict(x_context, y_context, x_query)
            preds[:, i] = y_pred
    return preds


def compute_gd_alignment(
    model,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    task_dim: int,
    gd_eta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    w_model = extract_implicit_weights(model, x_context, y_context, task_dim=task_dim)
    w_gd = gd_one_step_weights(x_context, y_context, eta=gd_eta)
    cos = cosine_similarity(w_model, w_gd)
    return {
        "w_model": w_model,
        "w_gd": w_gd,
        "cosine": cos,
    }

