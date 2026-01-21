import numpy as np

def mean_squared_error(pred, target):
    """
    Computes mean squared error

    Args:
        pred: predicted values
        target: ground truth values

    Returns:
        mse: scalar tensor
    """
    return ((pred - target) ** 2).mean()