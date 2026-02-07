import torch
from tqdm import tqdm

from model import TransformerModel, SparseTransformerModel
from train import MODEL_DIMS, MODEL_POSITIONS, MODEL_EMBED_DIM, MODEL_ATT_LAYERS, MODEL_ATT_HEADS, SPARSE_WINDOW_SIZE, SPARSE_STRIDE, SPARSE_GLOBAL_TOKENS
from data_sampler import sample_linear_regression_task


def load_model(state_path):
    """
    Loads a saved TransformerModel from a checkpoint

    Args:
        state_path: path to model checkpoint file

    Returns:
        model: TransformerModel loaded with checkpoint weights, in eval mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        n_dims = MODEL_DIMS,
        n_positions = MODEL_POSITIONS,
        n_embd = MODEL_EMBED_DIM,
        n_layer = MODEL_ATT_LAYERS,
        n_head = MODEL_ATT_HEADS,
    ).to(device)

    state = torch.load(state_path, map_location = device)
    state_dict = state["model_state_dict"]
    
    model.load_state_dict(state_dict, strict = False)
    model.eval()
    
    return model

def load_sparse_model(state_path):
    """
    Loads a saved SparseTransformerModel from a checkpoint.
    
    Args:
        state_path: path to model checkpoint file (.pt)

    Returns:
        model: SparseTransformerModel loaded with checkpoint weights, in eval mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SparseTransformerModel(
        n_dims = MODEL_DIMS,
        n_positions = MODEL_POSITIONS,
        n_embd = MODEL_EMBED_DIM,
        n_layer = MODEL_ATT_LAYERS,
        n_head = MODEL_ATT_HEADS,
        window_size = SPARSE_WINDOW_SIZE,
        stride = SPARSE_STRIDE,
        global_tokens = SPARSE_GLOBAL_TOKENS,
    ).to(device)

    state = torch.load(state_path, map_location = device)
    state_dict = state["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

@torch.no_grad()
def eval_model(model, n_points, n_dims, num_batches, batch_size, sparsity = None):
    """
    Evaluates a trained model on fresh linear-regression tasks

    Args:
        model: trained model
        n_points: number of (x, y) pairs per task (T)
        n_dims: full dimensionality of x (D)
        num_batches: number of batches to evaluate over
        batch_size: number of tasks per batch (B)
        sparsity: if not None, number of nonzero coordinates in w

    Returns:
        losses: [T] array of normalized mean squared errors across tasks
    """
    device = next(model.parameters()).device
    model.eval()

    denom = sparsity if sparsity else n_dims
    mses = 0.0 

    for _ in tqdm(range(num_batches)):
        xs, ys, w = sample_linear_regression_task(
            batch_size = batch_size,
            n_points = n_points,
            n_dims = n_dims,
            device = device,
            n_dims_trunc = n_dims,
            sparsity = sparsity
        )

        preds = model(xs, ys)
        sq_err = (preds - ys) ** 2 / denom

        mses += sq_err.sum(dim = 0)

    losses = (mses / (num_batches * batch_size)).cpu().numpy()

    return losses