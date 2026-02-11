"""Classical baselines for linear regression in-context prompts."""
from __future__ import annotations

from typing import Dict

import torch


def _solve_lstsq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: (n, d), y: (n,)
    return torch.linalg.lstsq(x, y).solution


def _solve_ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    d = x.size(-1)
    xtx = x.T @ x
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    return torch.linalg.solve(xtx + lam * eye, x.T @ y)


def predict_least_squares(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    batch_size = x_context.size(0)
    out = torch.zeros(batch_size, device=x_context.device)
    for b in range(batch_size):
        w = _solve_lstsq(x_context[b], y_context[b])
        out[b] = torch.dot(x_query[b, 0], w)
    return out


def predict_min_norm(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    # torch.linalg.lstsq returns the minimum-norm solution in underdetermined settings.
    return predict_least_squares(x_context, y_context, x_query)


def predict_ridge(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    batch_size = x_context.size(0)
    out = torch.zeros(batch_size, device=x_context.device)
    for b in range(batch_size):
        w = _solve_ridge(x_context[b], y_context[b], lam=lam)
        out[b] = torch.dot(x_query[b, 0], w)
    return out


def predict_knn(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    diff = x_context - x_query
    dist2 = (diff * diff).sum(dim=-1)  # (b, n)
    nn_idx = torch.topk(dist2, k=min(k, dist2.size(1)), dim=1, largest=False).indices
    gathered = torch.gather(y_context, dim=1, index=nn_idx)
    return gathered.mean(dim=1)


def predict_averaging(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    # One-step moment estimate: w_hat = (1/n) sum y_i x_i
    w_hat = torch.einsum("bn,bnd->bd", y_context, x_context) / x_context.size(1)
    return torch.einsum("bd,bd->b", w_hat, x_query[:, 0, :])


def predict_gd_one_step(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    # One-step GD from w0=0: w1 = (eta/n) * sum_i y_i x_i
    w1 = eta * torch.einsum("bn,bnd->bd", y_context, x_context) / x_context.size(1)
    return torch.einsum("bd,bd->b", w1, x_query[:, 0, :])


def run_all_baselines(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    knn_k: int = 3,
    ridge_lambda: float = 1e-2,
    gd_eta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    return {
        "least_squares": predict_least_squares(x_context, y_context, x_query),
        "min_norm": predict_min_norm(x_context, y_context, x_query),
        "ridge": predict_ridge(x_context, y_context, x_query, lam=ridge_lambda),
        "knn": predict_knn(x_context, y_context, x_query, k=knn_k),
        "averaging": predict_averaging(x_context, y_context, x_query),
        "gd_one_step": predict_gd_one_step(x_context, y_context, x_query, eta=gd_eta),
    }
