# scripts/plot_gqa_steps.py

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    metrics_path = "results/gqa_steps/metrics_steps.npz"
    data = np.load(metrics_path)

    steps = data["steps"]
    mse_model = data["mse_model"]
    mse_gd = data["mse_gd"]
    mse_oracle = data["mse_oracle"]
    cos_sim = data["cos_sim_to_gd"]

    out_dir = os.path.dirname(metrics_path)

    # Figure 1: Loss vs training steps 
    plt.figure(figsize=(5, 3))
    plt.plot(steps, mse_gd, label="GD", linewidth=2)
    plt.plot(steps, mse_model, label="Trained TF (GQA)", linewidth=2)
    plt.xlabel("Training steps")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gqa_steps_loss.png"), dpi=300)

    # Figure 2: diffs + cosine vs steps 
    model_diff = np.abs(mse_model - mse_oracle)
    preds_diff = np.abs(mse_model - mse_gd)

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax2 = ax1.twinx()

    ax1.plot(steps, preds_diff, label="Preds diff", linewidth=2, color="tab:orange")
    ax1.plot(steps, model_diff, label="Model diff", linewidth=2, color="tab:brown")
    ax2.plot(steps, cos_sim, label="Model cos", linewidth=2, color="tab:green")

    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("L2 / diff proxy")
    ax2.set_ylabel("Cosine sim")

    # combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gqa_steps_diff_cos.png"), dpi=300)


if __name__ == "__main__":
    main()