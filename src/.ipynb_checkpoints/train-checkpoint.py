import os
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm

from model import TransformerModel, SparseTransformerModel
from data_sampler import sample_linear_regression_task
from curriculum import Curriculum
from losses import mean_squared_error

# model hyperparameters
MODEL_DIMS = 20
MODEL_POSITIONS = 101 # must be > max points in curriculum
MODEL_EMBED_DIM = 256
MODEL_ATT_LAYERS = 12
MODEL_ATT_HEADS = 8

# sparse attention model hyperparameters
SPARSE_WINDOW_SIZE = 64
SPARSE_STRIDE = 16
SPARSE_GLOBAL_TOKENS = 0

# training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TRAIN_STEPS = 500_001
SAVE_EVERY_STEPS = 1_000 # interval to checkpoint
KEEP_EVERY_STEPS = 100_000 # interval to snapshot

@dataclass
class TrainConfig:
    out_dir: str

    sparsity: Optional[int]

    dims_start: int
    dims_end: int
    dims_inc: int

    points_start: int
    points_end: int
    points_inc: int
    
    interval: int

CONFIGS = {
    "underparameterized": TrainConfig(
        out_dir = "models/underparameterized_linear_regression",
        
        sparsity = None,

        dims_start = 5,
        dims_end = 20,
        dims_inc = 1,

        points_start = 11,
        points_end = 41,
        points_inc = 2,
        
        interval = 2_000,
    ),

    "overparameterized": TrainConfig(
        out_dir = "models/overparameterized_linear_regression",
        
        sparsity = None,

        dims_start = 20,
        dims_end = 20,
        dims_inc = 0,

        points_start = 5,
        points_end = 15,
        points_inc = 1,
        
        interval = 2_000,
    ),

    "sparse": TrainConfig(
        out_dir = "models/sparse_linear_regression",
        
        sparsity = 3, 

        dims_start = 5,
        dims_end = 20,
        dims_inc = 1,

        points_start = 11,
        points_end = 41,
        points_inc = 2,
        
        interval = 2_000,
    )  
}

def train(model, cfg):
    """
    Runs the full training loop for a linear-regression in-context learning task

    Args:
        model: TransformerModel instance
        cfg: TrainConfig specifying curriculum + sparsity settings
    """
    device = next(model.parameters()).device

    os.makedirs(cfg.out_dir, exist_ok = True)
    state_path = os.path.join(cfg.out_dir, "state.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    curriculum = Curriculum(
        dims_start = cfg.dims_start,
        dims_end = cfg.dims_end,
        dims_inc = cfg.dims_inc,
        points_start = cfg.points_start,
        points_end = cfg.points_end,
        points_inc = cfg.points_inc,
        interval = cfg.interval,
    )

    # check if model state has been checkpointed
    starting_step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

        # fast forward curriculum
        for _ in range(starting_step + 1):
            curriculum.update()

    # track an exponential moving average loss for easier interpretability
    ema_loss = None
    alpha = 0.01

    pbar = tqdm(range(starting_step, TRAIN_STEPS))
    for step in pbar:
        n_points = curriculum.points
        n_dims = curriculum.dims

        # sample data
        xs, ys, w = sample_linear_regression_task(
            batch_size = BATCH_SIZE,
            n_points = n_points,
            n_dims = MODEL_DIMS,
            device = device,
            n_dims_trunc = n_dims,
            sparsity = cfg.sparsity
        )

        # training step
        model.train()
        optimizer.zero_grad()
        preds = model(xs, ys)
        loss = mean_squared_error(preds, ys)
        loss.backward()
        optimizer.step()

        # ema loss
        loss_val = float(loss.detach().item())
        ema_loss = loss_val if ema_loss is None else alpha * loss_val + (1 - alpha) * ema_loss

        # update curriculum
        curriculum.update()

        # set progress bar description
        pbar.set_description(str({
            "loss": round(loss_val, 4),
            "ema": round(ema_loss, 4),
            "d": n_dims,
            "n": n_points,
        }))

        # checkpoint
        if step % SAVE_EVERY_STEPS == 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": step,
            }
            torch.save(training_state, state_path)

        # snapshot
        if step % KEEP_EVERY_STEPS == 0 and step > 0:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.out_dir, f"model_{step}.pt"),
            )    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        type = str,
        choices = list(CONFIGS.keys()),
        default = "underparameterized",
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)

    args = parse_args()
    cfg = CONFIGS[args.setting]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = TransformerModel(
    #     n_dims = MODEL_DIMS,
    #     n_positions = MODEL_POSITIONS,
    #     n_embd = MODEL_EMBED_DIM,
    #     n_layer = MODEL_ATT_LAYERS,
    #     n_head = MODEL_ATT_HEADS,
    # ).to(device)

    model = SparseTransformerModel(
        n_dims = MODEL_DIMS,
        n_positions = MODEL_POSITIONS,
        n_embd = MODEL_EMBED_DIM,
        n_layer = MODEL_ATT_LAYERS,
        n_head = MODEL_ATT_HEADS,
        window_size = SPARSE_WINDOW_SIZE,
        stride = SPARSE_STRIDE,
        global_tokens = SPARSE_GLOBAL_TOKENS,
        resid_pdrop = 0.0,
        attn_pdrop = 0.0,
    ).to(device)

    train(model, cfg)