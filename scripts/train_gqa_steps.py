# scripts/train_gqa_steps.py

import os
import numpy as np
import torch
import torch.nn.functional as F

from src.data.linear_regression import generate_linear_regression_batch
from src.evaluation.gd_baseline import gd_baseline_predict, batch_cosine_similarity
from src.models.gqa import GQATransformer


def evaluate(model, d, n, noise_std, device, batch_size=256, gd_steps=50, gd_lr=0.1):
    model.eval()

    # --- Generate eval batch (no grads needed for data) ---
    xs, ys, query_x, query_y, _ = generate_linear_regression_batch(
        batch_size=batch_size,
        d=d,
        n=n,
        noise_std=noise_std,
        device=device,
    )

    # --- Model predictions: we don't need grads for the model during eval ---
    with torch.no_grad():
        y_hat_model = model(xs, ys, query_x)  # [B]

    # --- GD baseline: we DO need grads for w, so NO no_grad() here ---
    y_hat_gd = gd_baseline_predict(xs, ys, query_x, gd_steps, gd_lr)

    # Oracle is just the true y
    y_hat_oracle = query_y

    mse_model = F.mse_loss(y_hat_model, query_y).item()
    mse_gd = F.mse_loss(y_hat_gd, query_y).item()
    mse_oracle = F.mse_loss(y_hat_oracle, query_y).item()

    cos_sim = batch_cosine_similarity(y_hat_model, y_hat_gd)

    return mse_model, mse_gd, mse_oracle, cos_sim

def main():
    # ---- hyperparameters: match Oswald-ish single-layer experiment ----
    d = 20
    n = 41                    # 2d + 1
    noise_std = 0.1
    num_layers = 8            # mid/high depth for this experiment
    num_q_heads = 4
    num_kv_heads = 1          # GQA: 4 Q heads sharing 1 KV group
    d_model = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_steps = 100_000       # 0 -> 100k training steps
    batch_size = 64
    lr = 1e-4

    # checkpoints for evaluation (we'll log at these steps)
    eval_steps = [0, 1000, 5000, 10_000, 25_000, 50_000, 75_000, 100_000]

    out_dir = "results/gqa_steps"
    os.makedirs(out_dir, exist_ok=True)

    # ---- model + optimizer ----
    model = GQATransformer(
        d=d,
        num_layers=num_layers,
        hidden_dim=d_model,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- containers for metrics ----
    steps_hist = []
    mse_model_hist = []
    mse_gd_hist = []
    mse_oracle_hist = []
    cos_hist = []

    # eval at step 0 (untrained model)
    mse_m, mse_g, mse_o, cos = evaluate(model, d, n, noise_std, device)
    steps_hist.append(0)
    mse_model_hist.append(mse_m)
    mse_gd_hist.append(mse_g)
    mse_oracle_hist.append(mse_o)
    cos_hist.append(cos)
    print(f"[step 0] mse_model={mse_m:.4f} mse_gd={mse_g:.4f} cos={cos:.4f}")

    # ---- training loop ----
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

        if step in eval_steps:
            mse_m, mse_g, mse_o, cos = evaluate(model, d, n, noise_std, device)
            steps_hist.append(step)
            mse_model_hist.append(mse_m)
            mse_gd_hist.append(mse_g)
            mse_oracle_hist.append(mse_o)
            cos_hist.append(cos)

            print(
                f"[step {step}] "
                f"loss={loss.item():.4f} | "
                f"mse_model={mse_m:.4f} mse_gd={mse_g:.4f} cos={cos:.4f}"
            )

    # ---- save metrics ----
    np.savez(
        os.path.join(out_dir, "metrics_steps.npz"),
        steps=np.array(steps_hist),
        mse_model=np.array(mse_model_hist),
        mse_gd=np.array(mse_gd_hist),
        mse_oracle=np.array(mse_oracle_hist),
        cos_sim_to_gd=np.array(cos_hist),
    )

    print(f"Saved metrics to {os.path.join(out_dir, 'metrics_steps.npz')}")


if __name__ == "__main__":
    main()