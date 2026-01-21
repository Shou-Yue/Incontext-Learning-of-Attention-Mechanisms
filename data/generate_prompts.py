import torch


def generate_batch(batch_size: int, dim: int, n_examples: int, noise_std: float = 0.0):
    """
    Generate a batch of linear regression prompts for in-context learning.
    
    Args:
        batch_size: Number of prompts in batch
        dim: Dimension of input/output space
        n_examples: Number of in-context examples per prompt
        noise_std: Standard deviation of label noise
    
    Returns:
        Dictionary containing:
        - inputs: (batch_size, 2*n_examples + 1, dim) - formatted prompt sequence
        - targets: (batch_size, 2*n_examples + 1, 1) - target values
        - w_star: (batch_size, dim) - ground truth weights (for analysis)
    """
    # Sample task-specific weight vectors
    w_star = torch.randn(batch_size, dim)
    
    # Sample in-context examples
    X = torch.randn(batch_size, n_examples, dim)
    y = torch.einsum('bnd,bd->bn', X, w_star)  # (batch, n_examples)
    
    # Add noise if specified
    if noise_std > 0:
        y += noise_std * torch.randn_like(y)
    
    # Sample query point
    x_query = torch.randn(batch_size, dim)
    y_query = torch.einsum('bd,bd->b', x_query, w_star)  # (batch,)
    
    # Format as sequence: (x1, y1), (x2, y2), ..., (xn, yn), x_query
    # Each token is d-dimensional, with y embedded in first position
    seq_len = 2 * n_examples + 1
    inputs = torch.zeros(batch_size, seq_len, dim)
    targets = torch.zeros(batch_size, seq_len, 1)
    
    for i in range(n_examples):
        # x_i token
        inputs[:, 2*i, :] = X[:, i, :]
        targets[:, 2*i, :] = 0  # Dummy target
        
        # y_i token (embedded as [y_i, 0, 0, ..., 0])
        inputs[:, 2*i + 1, 0] = y[:, i]
        targets[:, 2*i + 1, :] = 0  # Dummy target
    
    # Query token
    inputs[:, -1, :] = x_query
    targets[:, -1, 0] = y_query
    
    return {
        'inputs': inputs,
        'targets': targets,
        'w_star': w_star
    }


def get_curriculum_dim(step: int, start_dim: int, target_dim: int, total_steps: int) -> int:
    """
    Compute current dimension based on curriculum schedule.
    
    Args:
        step: Current training step
        start_dim: Starting dimension
        target_dim: Target dimension
        total_steps: Total curriculum steps
    
    Returns:
        Current dimension (int)
    """
    if step >= total_steps:
        return target_dim
    
    progress = step / total_steps
    current_dim = start_dim + (target_dim - start_dim) * progress
    return int(current_dim)
