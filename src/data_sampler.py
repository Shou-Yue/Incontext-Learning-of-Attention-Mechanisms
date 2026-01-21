import torch

def sample_gaussian_xs(batch_size, n_points, n_dims, n_dims_trunc, device):
    """
    Samples input vectors, x ~ N(0, I)

    Args:
        batch_size: number of tasks in the batch (B)
        n_points: number of (x, y) pairs per task (T)
        n_dims: full dimensionality of each x vector (D)
        n_dims_trunc: if not None and < D, zero out coordinates [n_dims_trunc:D]
        device: torch device to place tensors on

    Returns:
        xs: [B, T, D]
    """
    xs = torch.randn(batch_size, n_points, n_dims, device = device)
    
    if n_dims_trunc is not None and n_dims_trunc < n_dims:
        xs[:, :, n_dims_trunc:] = 0.0
        
    return xs

def sample_linear_regression_weights(batch_size, n_dims, device, sparsity = None):
    """
    Samples weight vectors, w ~ N(0, I)

    Args:
        batch_size: number of tasks in the batch (B)
        n_dims: dimensionality of each weight vector (D)
        device: torch device to place tensors on
        sparsity: if not None, number of active (nonzero) coordinates

    Returns:
        w: [B, D, 1]
    """
    w = torch.randn(batch_size, n_dims, 1, device = device)

    if sparsity is None or sparsity >= n_dims:
        return w

    # zero out entries at random if sparsity is specified
    for b in range(batch_size):
        idx = torch.randperm(n_dims, device = device)[:sparsity]
        mask = torch.zeros(n_dims, 1, device = device, dtype = torch.bool)
        mask[idx] = True
        w[b, ~mask] = 0.0

    return w

def evaluate_linear_regression(xs, w, scale = 1.0):
    """
    Computes linear outputs y = scale * (xᵀ w)

    Args:
        xs: [B, T, D] input vectors
        w: [B, D, 1] regression weights
        scale: optional multiplier applied to y

    Returns:
        ys: [B, T]
    """
    ys = scale * (xs @ w)
    return ys[:, :, 0]

def sample_linear_regression_task(batch_size, n_points, n_dims, device, n_dims_trunc = None, sparsity = None, scale = 1.0):
    """
    Samples a full linear regression task

    Args:
        batch_size: number of tasks in the batch (B)
        n_points: number of (x, y) pairs per task (T)
        n_dims: full dimensionality of each x vector (D)
        device: torch device to place tensors on
        n_dims_trunc: if not None, truncate x by zeroing dims [n_dims_trunc:D]
        sparsity: if not None, number of nonzero coordinates in w
        scale: optional multiplier applied to y = xᵀw

    Returns:
        xs: [B, T, D]
        ys: [B, T]
        w: [B, D, 1]
    """
    xs = sample_gaussian_xs(
        batch_size = batch_size,
        n_points = n_points,
        n_dims = n_dims,
        n_dims_trunc = n_dims_trunc,
        device = device,
    )

    w = sample_linear_regression_weights(
        batch_size = batch_size,
        n_dims = n_dims,
        device = device,
        sparsity = sparsity,
    )

    ys = evaluate_linear_regression(xs, w, scale = scale)

    return xs, ys, w