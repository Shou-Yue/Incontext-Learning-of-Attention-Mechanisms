# src/data/linear_regression.py

import torch


def generate_linear_regression_batch(
    batch_size: int,
    d: int,
    n: int,
    noise_std: float,
    device: str,
):
    """
    Generate a batch of linear regression in-context tasks.

    Each task b:
        w_b ~ N(0, I_d)
        x_{b,i} ~ N(0, I_d)
        y_{b,i} = x_{b,i}^T w_b + noise

    Returns:
        xs: [B, n, d]       in-context inputs
        ys: [B, n]          in-context targets
        query_x: [B, 1, d]  single query point per task
        query_y: [B]        ground truth at query_x
        w_true: [B, d]      underlying weight vector (for oracle if needed)
    """
    B = batch_size

    w_true = torch.randn(B, d, 1, device=device)               # [B, d, 1]

    xs = torch.randn(B, n, d, device=device)                   # [B, n, d]
    noise_ctx = noise_std * torch.randn(B, n, 1, device=device)
    ys = (xs @ w_true + noise_ctx).squeeze(-1)                 # [B, n]

    # one query point per task
    query_x = torch.randn(B, 1, d, device=device)              # [B, 1, d]
    noise_q = noise_std * torch.randn(B, 1, 1, device=device)
    query_y = (query_x @ w_true + noise_q).squeeze(-1).squeeze(-1)  # [B]

    w_true_flat = w_true.squeeze(-1)                           # [B, d]

    return xs, ys, query_x, query_y, w_true_flat