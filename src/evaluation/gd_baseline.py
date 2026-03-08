"""
Gradient Descent baselines for comparison with LSA.

Implements multi-step gradient descent on linear regression tasks.
"""
import torch
import numpy as np


def gd_one_step(xs, ys, query_x, eta=0.1):
    """
    One-step gradient descent from W0 = 0.
    
    W1 = W0 - eta * (1/N) * sum_i (W0 x_i - y_i) x_i^T
       = 0 - eta * (1/N) * sum_i (-y_i) x_i^T
       = eta * (1/N) * sum_i y_i x_i^T
    
    Args:
        xs: (batch_size, n_points, d) training inputs
        ys: (batch_size, n_points) training outputs
        query_x: (batch_size, 1, d) query input
        eta: learning rate
    
    Returns:
        y_pred: (batch_size,) predicted output
        delta_w: (batch_size, d) weight update W1 - W0 = W1
    """
    batch_size, n_points, d = xs.shape
    
    # W0 = 0
    W = torch.zeros(batch_size, d, device=xs.device)
    
    # Compute gradient: (1/N) * sum_i (W x_i - y_i) x_i^T
    # Residuals: W x_i - y_i
    predictions = torch.bmm(xs, W.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_points)
    residuals = predictions - ys  # (batch_size, n_points)
    
    # Gradient: (1/N) * sum_i residual_i * x_i
    gradient = torch.bmm(residuals.unsqueeze(1), xs).squeeze(1) / n_points  # (batch_size, d)
    
    # Update: W1 = W0 - eta * gradient
    W = W - eta * gradient
    
    # Prediction on query
    query_x_flat = query_x.squeeze(1)  # (batch_size, d)
    y_pred = (W * query_x_flat).sum(dim=1)  # (batch_size,)
    
    return y_pred, W


def gd_t_steps(xs, ys, query_x, eta=0.1, T=1):
    """
    T-step gradient descent from W0 = 0.
    
    For t = 0 to T-1:
        W_{t+1} = W_t - eta * (1/N) * sum_i (W_t x_i - y_i) x_i^T
    
    Args:
        xs: (batch_size, n_points, d) training inputs
        ys: (batch_size, n_points) training outputs
        query_x: (batch_size, 1, d) query input
        eta: learning rate
        T: number of gradient descent steps
    
    Returns:
        y_pred: (batch_size,) predicted output after T steps
        delta_w: (batch_size, d) final weight W_T
    """
    batch_size, n_points, d = xs.shape
    
    # Initialize W0 = 0
    W = torch.zeros(batch_size, d, device=xs.device)
    
    # Perform T steps of gradient descent
    for t in range(T):
        # Compute predictions: W x_i
        predictions = torch.bmm(xs, W.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_points)
        
        # Residuals: W x_i - y_i
        residuals = predictions - ys  # (batch_size, n_points)
        
        # Gradient: (1/N) * sum_i residual_i * x_i
        gradient = torch.bmm(residuals.unsqueeze(1), xs).squeeze(1) / n_points  # (batch_size, d)
        
        # Update: W_{t+1} = W_t - eta * gradient
        W = W - eta * gradient
    
    # Prediction on query
    query_x_flat = query_x.squeeze(1)  # (batch_size, d)
    y_pred = (W * query_x_flat).sum(dim=1)  # (batch_size,)
    
    return y_pred, W


def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two sets of vectors.
    
    Args:
        vec1: (batch_size, d) first set of vectors
        vec2: (batch_size, d) second set of vectors
    
    Returns:
        cos_sim: (batch_size,) cosine similarities
    """
    # Normalize vectors
    vec1_norm = vec1 / (torch.norm(vec1, dim=1, keepdim=True) + 1e-8)
    vec2_norm = vec2 / (torch.norm(vec2, dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity
    cos_sim = (vec1_norm * vec2_norm).sum(dim=1)
    
    return cos_sim


def generate_linear_regression_task(d, n_points, batch_size=1, sigma=0.0, device='cpu'):
    """
    Generate a linear regression task with Gaussian inputs and labels.
    
    Args:
        d: input dimension
        n_points: number of training points (typically n = 2d + 1)
        batch_size: number of tasks to generate
        sigma: label noise level (std dev)
        device: torch device
    
    Returns:
        xs: (batch_size, n_points, d) training inputs
        ys: (batch_size, n_points) training outputs
        w_true: (batch_size, d) true weight vectors
    """
    # Sample true weights from N(0, I)
    w_true = torch.randn(batch_size, d, device=device)
    
    # Sample inputs from N(0, I)
    xs = torch.randn(batch_size, n_points, d, device=device)
    
    # Compute labels: y = w^T x + noise
    ys = torch.bmm(xs, w_true.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_points)
    
    # Add label noise
    if sigma > 0:
        noise = torch.randn_like(ys) * sigma
        ys = ys + noise
    
    return xs, ys, w_true


def evaluate_on_task(model, xs, ys, query_x, query_y, eta=0.1):
    """
    Evaluate LSA model and GD baseline on a single task.
    
    Args:
        model: MultiLayerLSA model
        xs: (batch_size, n_points, d) training inputs
        ys: (batch_size, n_points) training outputs
        query_x: (batch_size, 1, d) query input
        query_y: (batch_size,) true query output
        eta: learning rate for GD
    
    Returns:
        results: dict with MSE and cosine similarity metrics
    """
    device = xs.device
    T = model.num_layers
    
    with torch.no_grad():
        # LSA prediction
        y_pred_lsa = model(xs, ys, query_x)
        mse_lsa = ((y_pred_lsa - query_y) ** 2).mean().item()
        
        # GD prediction
        y_pred_gd, w_gd = gd_t_steps(xs, ys, query_x, eta=eta, T=T)
        mse_gd = ((y_pred_gd - query_y) ** 2).mean().item()
        
        # Get weight updates
        w_lsa = model.get_weight_update(xs, ys, query_x)
        
        # Cosine similarity between LSA and GD weight updates
        cos_sim = compute_cosine_similarity(w_lsa, w_gd).mean().item()
    
    return {
        'mse_lsa': mse_lsa,
        'mse_gd': mse_gd,
        'cosine_sim': cos_sim
    }
