import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Lasso

from data_sampler import sample_linear_regression_task

def least_squares_baseline(samples, n_points, n_dims, sparsity = None):
    """
    Computes the least-squares (or min-norm least squares) baseline
    for linear regression tasks

    Args:
        samples: number of independent tasks to average over
        n_points: number of (x, y) pairs per task (T)
        n_dims: full dimensionality of x (D)
        sparsity: if not None, number of nonzero coordinates in w

    Returns:
        lsq_mse: [T] array of normalized mean squared errors across tasks
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    losses = []
    denom = sparsity if sparsity else n_dims
    
    for _ in tqdm(range(samples)):
        xs, ys, w = sample_linear_regression_task(
            batch_size = 1,
            n_points = n_points,
            n_dims = n_dims,
            device = device,
            n_dims_trunc = n_dims,
            sparsity = sparsity
        )
        xs, ys = xs[0], ys[0]

        mses = []
        for i in range(len(xs)):
            X, y = xs[:i], ys[:i]
            x_query = xs[i].cpu().numpy()
            y_query = ys[i].cpu().numpy()
            
            # min-norm least squares estimator
            w_hat = np.linalg.lstsq(X, y, rcond = None)[0]
            y_pred = x_query @ w_hat

            # calculate mse divided by the number of nonzero weights
            mse = (y_pred - y_query) ** 2 / denom
            mses.append(mse)
        
        losses.append(mses)
    
    return np.mean(losses, axis = 0)

def lasso_baseline(samples, n_points, n_dims, sparsity = None):
    """
    Computes the LASSO baseline for sparse linear regression tasks.

    Args:
        samples: number of independent tasks to average over
        n_points: number of (x, y) pairs per task (T)
        n_dims: full dimensionality of x (D)
        sparsity: if not None, number of nonzero coordinates in w

    Returns:
        lasso_mse: [T - 1] array of normalized mean squared errors across tasks
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    losses = []
    denom = sparsity if sparsity else n_dims
    
    for _ in tqdm(range(samples)):
        xs, ys, w = sample_linear_regression_task(
            batch_size = 1,
            n_points = n_points,
            n_dims = n_dims,
            device = device,
            n_dims_trunc = n_dims,
            sparsity = sparsity
        )
        xs, ys = xs[0], ys[0]

        mses = []
        for i in range(1, len(xs)):
            X, y = xs[:i], ys[:i]
            x_query = xs[i].cpu().numpy()
            y_query = ys[i].cpu().numpy()

            # convert to numpy for sklearn
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy().reshape(-1)
            
            # lasso estimator
            lasso = Lasso(
                alpha = 0.1,
                fit_intercept = False,
                max_iter = 10_000
            )
            lasso.fit(X_np, y_np)
            y_pred = lasso.predict(x_query[None, :])[0]

            # calculate mse divided by the number of nonzero weights
            mse = (y_pred - y_query) ** 2 / denom
            mses.append(mse)
        
        losses.append(mses)
    
    return np.mean(losses, axis = 0)