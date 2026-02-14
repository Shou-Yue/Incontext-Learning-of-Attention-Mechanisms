"""
Baseline methods for comparison with transformer.
"""
import torch


def least_squares_solution(xs, ys):
    """
    Compute the least squares solution for linear regression.
    For overparameterized case (n_points > n_dims), this is the optimal solution.
    
    Solves: w* = argmin_w ||Xw - y||^2
    Solution: w* = (X^T X)^{-1} X^T y (or pseudoinverse when singular)
    
    Args:
        xs: Input tensor of shape (batch_size, n_points, n_dims)
        ys: Output tensor of shape (batch_size, n_points)
    
    Returns:
        w_hat: Estimated weight vectors of shape (batch_size, n_dims, 1)
    """
    batch_size, n_points, n_dims = xs.shape
    
    # Reshape for batch matrix operations
    # xs: (batch_size, n_points, n_dims)
    # ys: (batch_size, n_points, 1)
    ys_expanded = ys.unsqueeze(-1)
    
    # Compute w = (X^T X)^{-1} X^T y using pseudoinverse
    # This handles both over and underparameterized cases
    try:
        # Using lstsq for numerical stability
        w_hat = torch.linalg.lstsq(xs, ys_expanded).solution
    except:
        # Fallback to pseudoinverse
        xs_pinv = torch.linalg.pinv(xs)  # (batch_size, n_dims, n_points)
        w_hat = xs_pinv @ ys_expanded     # (batch_size, n_dims, 1)
    
    return w_hat


def ridge_regression_solution(xs, ys, lambda_reg=0.0):
    """
    Compute the ridge regression solution.
    
    Solves: w* = argmin_w ||Xw - y||^2 + lambda * ||w||^2
    Solution: w* = (X^T X + lambda * I)^{-1} X^T y
    
    Args:
        xs: Input tensor of shape (batch_size, n_points, n_dims)
        ys: Output tensor of shape (batch_size, n_points)
        lambda_reg: Regularization parameter
    
    Returns:
        w_hat: Estimated weight vectors of shape (batch_size, n_dims, 1)
    """
    batch_size, n_points, n_dims = xs.shape
    ys_expanded = ys.unsqueeze(-1)
    
    # Compute X^T X
    xtx = xs.transpose(1, 2) @ xs  # (batch_size, n_dims, n_dims)
    
    # Add regularization: X^T X + lambda * I
    identity = torch.eye(n_dims, device=xs.device).unsqueeze(0)
    xtx_reg = xtx + lambda_reg * identity
    
    # Compute (X^T X + lambda * I)^{-1} X^T y
    xty = xs.transpose(1, 2) @ ys_expanded  # (batch_size, n_dims, 1)
    w_hat = torch.linalg.solve(xtx_reg, xty)
    
    return w_hat


def predict_from_weights(xs_test, w_hat):
    """
    Make predictions using estimated weights.
    
    Args:
        xs_test: Test inputs of shape (batch_size, n_test_points, n_dims)
        w_hat: Estimated weights of shape (batch_size, n_dims, 1)
    
    Returns:
        y_pred: Predictions of shape (batch_size, n_test_points)
    """
    y_pred = (xs_test @ w_hat).squeeze(-1)
    return y_pred


def knn_prediction(xs_train, ys_train, xs_test, k=3):
    """
    k-Nearest Neighbors prediction.
    
    Args:
        xs_train: Training inputs of shape (batch_size, n_train_points, n_dims)
        ys_train: Training outputs of shape (batch_size, n_train_points)
        xs_test: Test inputs of shape (batch_size, n_test_points, n_dims)
        k: Number of nearest neighbors
    
    Returns:
        y_pred: Predictions of shape (batch_size, n_test_points)
    """
    batch_size, n_test, n_dims = xs_test.shape
    n_train = xs_train.shape[1]
    
    # Compute pairwise distances
    # xs_test: (batch_size, n_test, n_dims)
    # xs_train: (batch_size, n_train, n_dims)
    # distances: (batch_size, n_test, n_train)
    distances = torch.cdist(xs_test, xs_train, p=2)
    
    # Find k nearest neighbors
    # topk_indices: (batch_size, n_test, k)
    _, topk_indices = torch.topk(distances, k=min(k, n_train), largest=False, dim=-1)
    
    # Gather corresponding y values
    # Expand ys_train for gathering: (batch_size, 1, n_train) -> (batch_size, n_test, n_train)
    ys_train_expanded = ys_train.unsqueeze(1).expand(-1, n_test, -1)
    
    # Gather k nearest y values: (batch_size, n_test, k)
    topk_ys = torch.gather(ys_train_expanded, dim=-1, index=topk_indices)
    
    # Average the k nearest neighbors
    y_pred = topk_ys.mean(dim=-1)
    
    return y_pred


def averaging_prediction(xs_train, ys_train, xs_test):
    """
    Simple averaging baseline - predict mean of all training outputs.
    
    Args:
        xs_train: Training inputs of shape (batch_size, n_train_points, n_dims)
        ys_train: Training outputs of shape (batch_size, n_train_points)
        xs_test: Test inputs of shape (batch_size, n_test_points, n_dims)
    
    Returns:
        y_pred: Predictions of shape (batch_size, n_test_points)
    """
    # Compute mean of training outputs for each batch
    y_mean = ys_train.mean(dim=1, keepdim=True)  # (batch_size, 1)
    
    # Broadcast to test points
    n_test = xs_test.shape[1]
    y_pred = y_mean.expand(-1, n_test)  # (batch_size, n_test_points)
    
    return y_pred


def evaluate_baseline(model, xs_train, ys_train, xs_test, ys_test, method='least_squares'):
    """
    Evaluate a baseline method on test data.
    
    Args:
        model: Not used for baselines, kept for API consistency
        xs_train: Training inputs of shape (batch_size, n_train_points, n_dims)
        ys_train: Training outputs of shape (batch_size, n_train_points)
        xs_test: Test inputs of shape (batch_size, n_test_points, n_dims)
        ys_test: Test outputs of shape (batch_size, n_test_points)
        method: One of 'least_squares', 'ridge', 'knn', 'averaging'
    
    Returns:
        mse: Mean squared error on test set
    """
    if method == 'least_squares':
        w_hat = least_squares_solution(xs_train, ys_train)
        y_pred = predict_from_weights(xs_test, w_hat)
    elif method == 'ridge':
        w_hat = ridge_regression_solution(xs_train, ys_train, lambda_reg=0.01)
        y_pred = predict_from_weights(xs_test, w_hat)
    elif method == 'knn':
        y_pred = knn_prediction(xs_train, ys_train, xs_test, k=3)
    elif method == 'averaging':
        y_pred = averaging_prediction(xs_train, ys_train, xs_test)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute MSE
    mse = ((y_pred - ys_test) ** 2).mean()
    
    return mse.item()
