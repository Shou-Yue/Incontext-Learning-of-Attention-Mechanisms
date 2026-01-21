import torch
import torch.nn.functional as F


def compute_mse(predictions, targets):
    """
    Compute mean squared error.
    
    Args:
        predictions: (batch, seq_len, 1) or (batch,)
        targets: (batch, seq_len, 1) or (batch,)
    
    Returns:
        mse: scalar
    """
    return F.mse_loss(predictions, targets).item()


def extract_implicit_weights(model, X, y, x_test, device='cpu'):
    """
    Extract the model's implicit weight vector by testing on canonical basis vectors.
    
    Args:
        model: Trained transformer
        X: (batch, n_examples, dim) - in-context examples
        y: (batch, n_examples) - in-context labels
        x_test: (batch, n_test, dim) - test inputs to probe model
        device: torch device
    
    Returns:
        w_model: (batch, dim) - extracted weight vectors
    """
    batch_size, n_examples, dim = X.shape
    n_test = x_test.shape[1]
    
    # Format prompt: (x1, y1), (x2, y2), ..., (xn, yn), x_test
    seq_len = 2 * n_examples + 1
    inputs = torch.zeros(batch_size, seq_len, dim).to(device)
    
    for i in range(n_examples):
        inputs[:, 2*i, :] = X[:, i, :]
        inputs[:, 2*i + 1, 0] = y[:, i]
    
    # We'll probe one test point at a time
    predictions = []
    for i in range(n_test):
        inputs[:, -1, :] = x_test[:, i, :]
        with torch.no_grad():
            pred = model(inputs)[:, -1, 0]  # (batch,)
        predictions.append(pred)
    
    # Stack predictions: (batch, n_test)
    predictions = torch.stack(predictions, dim=1)
    
    # Solve for weights: X_test @ w ≈ predictions
    # w = (X_test^T X_test)^{-1} X_test^T predictions
    w_model = torch.linalg.lstsq(x_test, predictions.unsqueeze(-1)).solution.squeeze(-1)
    
    return w_model


def compute_cosine_similarity_to_gd(model, batch, device='cpu'):
    """
    Compute cosine similarity between model's implicit weights and GD-1 baseline.
    
    Args:
        model: Trained transformer
        batch: Dictionary from generate_batch containing:
            - inputs: (batch_size, seq_len, dim)
            - w_star: (batch_size, dim)
        device: torch device
    
    Returns:
        cosine_sim: scalar (average over batch)
    """
    inputs = batch['inputs'].to(device)
    w_star = batch['w_star'].to(device)
    batch_size, seq_len, dim = inputs.shape
    
    # Extract in-context examples from inputs
    n_examples = (seq_len - 1) // 2
    X = torch.zeros(batch_size, n_examples, dim).to(device)
    y = torch.zeros(batch_size, n_examples).to(device)
    
    for i in range(n_examples):
        X[:, i, :] = inputs[:, 2*i, :]
        y[:, i] = inputs[:, 2*i + 1, 0]
    
    # Compute GD-1 weights: w_GD = (1/n) X^T y
    # (Gradient descent from w=0 with step size 1)
    w_GD = torch.einsum('bnd,bn->bd', X, y) / n_examples  # (batch, dim)
    
    # Extract model's implicit weights using canonical basis
    x_test = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # (batch, dim, dim)
    w_model = extract_implicit_weights(model, X, y, x_test, device)  # (batch, dim)
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(w_model, w_GD, dim=-1)  # (batch,)
    
    return cos_sim.mean().item()


def evaluate_model(model, config, num_batches=10, device='cpu'):
    """
    Evaluate model on held-out prompts.
    
    Args:
        model: Trained transformer
        config: ExperimentConfig
        num_batches: Number of evaluation batches
        device: torch device
    
    Returns:
        Dictionary of metrics
    """
    from data.generate_prompts import generate_batch
    
    model.eval()
    total_mse = 0.0
    total_cos_sim = 0.0
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate evaluation batch
            batch = generate_batch(
                batch_size=config.batch_size,
                dim=config.task_dim,
                n_examples=config.max_prompt_length // 2,
                noise_std=config.noise_std
            )
            
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            
            # Compute predictions
            predictions = model(inputs)
            
            # MSE on query token only (last token)
            query_pred = predictions[:, -1, :]
            query_target = targets[:, -1, :]
            total_mse += compute_mse(query_pred, query_target)
            
            # Cosine similarity
            total_cos_sim += compute_cosine_similarity_to_gd(model, batch, device)
    
    return {
        'mse': total_mse / num_batches,
        'cosine_sim': total_cos_sim / num_batches
    }
