import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import TransformerModel
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
SPARSE_WINDOW_SIZE = 16
SPARSE_STRIDE = 16
SPARSE_GLOBAL_TOKENS = 0

# training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TRAIN_STEPS = 500_001

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
    resid_pdrop = 0.0,
    attn_pdrop = 0.0,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# initialize curriculum
curriculum = Curriculum(
    dims_start = 5, dims_end = 20, dims_inc = 1,
    points_start = 11, points_end = 41, points_inc = 2,
    interval = 2_000
)

results = []
pbar = tqdm(range(TRAIN_STEPS))

# track an exponential moving average loss for easier interpretability
ema_loss = None
alpha = 0.01

for step in pbar:
    # sample data
    xs, ys, w = sample_linear_regression_task(
        batch_size = BATCH_SIZE,
        n_points = curriculum.points,
        n_dims = MODEL_DIMS,
        device = device,
        n_dims_trunc = curriculum.dims
    )

    # training step
    model.train()
    optimizer.zero_grad()
    preds = model(xs, ys)
    loss = mean_squared_error(preds, ys)
    loss.backward()
    optimizer.step()

    # update curriculum
    curriculum.update()

    # ema loss
    loss_val = float(loss.detach().item())
    ema_loss = loss_val if ema_loss is None else alpha * loss_val + (1 - alpha) * ema_loss

    # set progress bar description
    pbar.set_description(str({
        "loss": round(loss_val, 4),
        "ema": round(ema_loss, 4),
        "d": n_dims,
        "n": n_points,
    }))

    if step % 100 == 0:
        loss_val = loss.item()
        results.append({
            "step": step,
            "loss": loss_val,
            "dims": curriculum.dims,
            "points": curriculum.points
        })

df = pd.DataFrame(results)
df.to_csv("training_results.csv", index = False)
print("Results saved to training_results.csv")