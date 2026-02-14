"""
Task definitions for in-context learning.
"""
import torch


class LinearRegressionTask:
    """
    Linear regression task: f(x) = w^T x
    Weight vectors w sampled from N(0, I_d)
    """
    
    def __init__(self, n_dims, batch_size, n_dims_truncated=None, device='cpu'):
        """
        Args:
            n_dims: Ambient dimension
            batch_size: Number of functions to sample
            n_dims_truncated: If specified, zero out weight dimensions beyond this (for curriculum)
            device: Device to create tensors on
        """
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.device = device
        
        # Sample weight vectors from N(0, I_d)
        self.w = torch.randn(batch_size, n_dims, 1, device=device)
        
        # Zero out dimensions beyond n_dims_truncated (for curriculum learning)
        if n_dims_truncated is not None and n_dims_truncated < n_dims:
            self.w[:, n_dims_truncated:, :] = 0
    
    def evaluate(self, xs):
        """
        Evaluate f(x) = w^T x for all x in xs.
        
        Args:
            xs: Input tensor of shape (batch_size, n_points, n_dims)
        
        Returns:
            ys: Output tensor of shape (batch_size, n_points)
        """
        # xs @ w gives shape (batch_size, n_points, 1), we squeeze the last dim
        ys = (xs @ self.w).squeeze(-1)
        return ys


def squared_error(y_pred, y_true):
    """Squared error loss (element-wise)."""
    return (y_pred - y_true) ** 2


def mean_squared_error(y_pred, y_true):
    """Mean squared error loss."""
    return ((y_pred - y_true) ** 2).mean()
