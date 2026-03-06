# scripts/plot_gqa_context.py

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_path = "results/gqa_context/metrics_context.npz"
    data = np.load(data_path)

    n = data["n"]
    mse_model = data["mse_model"]
    mse_gd = data["mse_gd"]
    mse_oracle = data["mse_oracle"]
    cos = data["cos_sim_to_gd"]

    out_dir = os.path.dirname(data_path)

    # --- Plot 1: MSE vs n ---
    plt.figure(figsize=(6, 4))
    plt.plot(n, mse_model, marker="o", label="GQA")
    plt.plot(n, mse_gd, marker="o", label="GD baseline")
    plt.plot(n, mse_oracle, marker="o", label="Oracle")
    plt.xlabel("Number of in-context examples (n)")
    plt.ylabel("MSE")
    plt.title("GQA vs GD vs Oracle (Context Length Sweep)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gqa_context_mse.png"), dpi=200)
    plt.close()

    # --- Plot 2: Cosine vs n ---
    plt.figure(figsize=(6, 4))
    plt.plot(n, cos, marker="o", label="cos(GQA, GD)")
    plt.xlabel("Number of in-context examples (n)")
    plt.ylabel("Cosine similarity")
    plt.title("Cosine Similarity to GD (Context Length Sweep)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gqa_context_cosine.png"), dpi=200)
    plt.close()

    print("Saved context sweep plots to:", out_dir)


if __name__ == "__main__":
    main()