# src/evaluation/gd_baseline.py

import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F


def gd_baseline_predict(
    xs: torch.Tensor,
    ys: torch.Tensor,
    query_x: torch.Tensor,
    num_steps: int = 50,
    lr: float = 0.1,
):
    """
    Simple per-task gradient descent baseline for linear regression.

    xs: [B, n, d]
    ys: [B, n]
    query_x: [B, 1, d]

    Returns:
        y_hat_gd: [B]
    """
    B, n, d = xs.shape
    device = xs.device

    # w: [B, d] with grads
    w = torch.zeros(B, d, device=device, requires_grad=True)

    for _ in range(num_steps):
        # predictions on context: [B, n]
        y_pred = torch.einsum("bnd,bd->bn", xs, w)

        loss = F.mse_loss(y_pred, ys)

        grad, = torch.autograd.grad(loss, w)

        with torch.no_grad():
            w -= lr * grad
        w.requires_grad_(True)

    with torch.no_grad():
        y_hat = torch.einsum("bnd,bd->bn", query_x, w)  # [B, 1]
        y_hat = y_hat[:, 0]  # [B]

    return y_hat


def batch_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Cosine similarity between a and b over the batch of predictions.

    a, b: [B] or [B, ...]
    Returns a scalar (mean over batch).
    """
    a_flat = a.view(a.size(0), -1)
    b_flat = b.view(b.size(0), -1)

    dot = (a_flat * b_flat).sum(dim=1)
    a_norm = a_flat.norm(dim=1)
    b_norm = b_flat.norm(dim=1)
    cos = dot / (a_norm * b_norm + eps)
    return cos.mean().item()