# scripts/plot_gqa_layers.py

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_path = "results/gqa_layers/metrics_layers.npz"
    data = np.load(data_path)

    num_layers = data["num_layers"]
    mse_model = data["mse_model"]
    mse_gd = data["mse_gd"]
    mse_oracle = data["mse_oracle"]
    cos = data["cos_sim_to_gd"]

    out_dir = os.path.dirname(data_path)

    # --- Plot 1: MSE vs num_layers ---
    plt.figure(figsize=(6, 4))
    plt.plot(num_layers, mse_model, marker="o", label="GQA")
    plt.plot(num_layers, mse_gd, marker="o", label="GD baseline")
    plt.plot(num_layers, mse_oracle, marker="o", label="Oracle")
    plt.xlabel("Number of layers")
    plt.ylabel("MSE")
    plt.title("GQA vs GD vs Oracle (Layer Sweep)")
    plt.xscale("log", base=2)  # 2,4,8,16,... looks nicer on log2
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gqa_layers_mse.png"), dpi=200)
    plt.close()

    # --- Plot 2: Cosine vs num_layers ---
    plt.figure(figsize=(6, 4))
    plt.plot(num_layers, cos, marker="o", label="cos(GQA, GD)")
    plt.xlabel("Number of layers")
    plt.ylabel("Cosine similarity")
    plt.title("Cosine Similarity to GD (Layer Sweep)")
    plt.xscale("log", base=2)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gqa_layers_cosine.png"), dpi=200)
    plt.close()

    print("Saved layer sweep plots to:", out_dir)


if __name__ == "__main__":
    main()