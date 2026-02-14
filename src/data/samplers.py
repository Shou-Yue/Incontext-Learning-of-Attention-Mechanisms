"""
Data sampling utilities for generating training prompts.
"""
import torch


class GaussianSampler:
    """Samples inputs from isotropic Gaussian distribution N(0, I_d)."""
    
    def __init__(self, n_dims):
        self.n_dims = n_dims
    
    def sample_xs(self, n_points, batch_size, n_dims_truncated=None, device='cpu'):
        """
        Sample input points from N(0, I_d).
        
        Args:
            n_points: Number of points per batch
            batch_size: Batch size
            n_dims_truncated: If specified, zero out dimensions beyond this (for curriculum)
            device: Device to create tensors on
        
        Returns:
            xs: Tensor of shape (batch_size, n_points, n_dims)
        """
        xs = torch.randn(batch_size, n_points, self.n_dims, device=device)
        
        # Zero out dimensions beyond n_dims_truncated (for curriculum learning)
        if n_dims_truncated is not None and n_dims_truncated < self.n_dims:
            xs[:, :, n_dims_truncated:] = 0
        
        return xs
