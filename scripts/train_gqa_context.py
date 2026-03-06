# scripts/train_gqa_context.py

import os
import numpy as np
import torch
import torch.nn.functional as F

from src.data.linear_regression import generate_linear_regression_batch
from src.evaluation.gd_baseline import gd_baseline_predict, batch_cosine_similarity
from src.models.gqa import GQATransformer


def evaluate(model, d, n, noise_std, device, batch_size=256, gd_steps=50, gd_lr=0.1):
    model.eval()

    xs, ys, query_x, query_y, _ = generate_linear_regression_batch(
        batch_size=batch_size,
        d=d,
        n=n,
        noise_std=noise_std,
        device=device,
    )

    with torch.no_grad():
        y_hat_model = model(xs, ys, query_x)

    y_hat_gd = gd_baseline_predict(xs, ys, query_x, gd_steps, gd_lr)
    y_hat_oracle = query_y

    mse_model = F.mse_loss(y_hat_model, query_y).item()
    mse_gd = F.mse_loss(y_hat_gd, query_y).item()
    mse_oracle = F.mse_loss(y_hat_oracle, query_y).item()
    cos_sim = batch_cosine_similarity(y_hat_model, y_hat_gd)

    return mse_model, mse_gd, mse_oracle, cos_sim


def train_for_context(n, d, noise_std, device,
                      num_steps=50000, batch_size=64,
                      lr=1e-4, num_layers=8,
                      num_q_heads=4, num_kv_heads=1):
    """Train GQA with given context length n, then evaluate."""
    model = GQATransformer(
        d=d,
        num_layers=num_layers,
        hidden_dim=64,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, num_steps + 1):
        model.train()
        xs, ys, query_x, query_y, _ = generate_linear_regression_batch(
            batch_size=batch_size,
            d=d,
            n=n,
            noise_std=noise_std,
            device=device,
        )

        y_hat = model(xs, ys, query_x)
        loss = F.mse_loss(y_hat, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mse_m, mse_g, mse_o, cos = evaluate(model, d, n, noise_std, device)
    return mse_m, mse_g, mse_o, cos


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = 20
    noise_std = 0.1

    # context lengths to try
    n_list = [11, 21, 41, 61, 81]

    mse_model_list = []
    mse_gd_list = []
    mse_oracle_list = []
    cos_list = []

    out_dir = "results/gqa_context"
    os.makedirs(out_dir, exist_ok=True)

    for n in n_list:
        print(f"=== Training GQA with n = {n} in-context examples ===")
        mse_m, mse_g, mse_o, cos = train_for_context(
            n=n,
            d=d,
            noise_std=noise_std,
            device=device,
            num_steps=50000,  # fixed training steps for this experiment
        )

        mse_model_list.append(mse_m)
        mse_gd_list.append(mse_g)
        mse_oracle_list.append(mse_o)
        cos_list.append(cos)

        print(
            f"[n={n}] mse_model={mse_m:.4f} "
            f"mse_gd={mse_g:.4f} mse_oracle={mse_o:.4f} cos={cos:.4f}"
        )

    np.savez(
        os.path.join(out_dir, "metrics_context.npz"),
        n=np.array(n_list),
        mse_model=np.array(mse_model_list),
        mse_gd=np.array(mse_gd_list),
        mse_oracle=np.array(mse_oracle_list),
        cos_sim_to_gd=np.array(cos_list),
    )

    print(f"Saved metrics to {os.path.join(out_dir, 'metrics_context.npz')}")


if __name__ == "__main__":
    main()