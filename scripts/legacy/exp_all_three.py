#!/usr/bin/env python3
"""
Master runner: training-steps sweep, layer sweep, and in-context sweep.

Uses:
  - exp_steps_sweep.py
  - lsa_gd_multilayer.py
  - exp_context_sweep.py
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run all three experiments")
    parser.add_argument('--output_root', type=str, default='./results',
                        help='Root output directory')
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto')

    # Shared model/lowrank args
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['lsa', 'softmax', 'linformer', 'kernel'])
    parser.add_argument('--lowrank_k_ratios', type=float, nargs='+', default=[1.0, 0.75, 0.5])
    parser.add_argument('--lowrank_block_size', type=int, default=2)

    # Steps sweep args
    parser.add_argument('--steps_n_points', type=int, default=41)
    parser.add_argument('--steps_num_layers', type=int, default=8)
    parser.add_argument('--train_steps_list', type=int, nargs='+',
                        default=[0, 1000, 2000, 5000, 10000, 20000])

    # Layer sweep args
    parser.add_argument('--num_layers_list', type=int, nargs='+',
                        default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--num_train_tasks', type=int, default=64000)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_eval_tasks', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.0)

    # Context sweep args
    parser.add_argument('--n_points_list', type=int, nargs='+',
                        default=[5, 10, 20, 40])
    parser.add_argument('--context_train_steps', type=int, default=10000)

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1) Training steps sweep
    run_cmd([
        sys.executable, "scripts/exp_steps_sweep.py",
        "--d", str(args.d),
        "--n_points", str(args.steps_n_points),
        "--num_layers", str(args.steps_num_layers),
        "--train_steps_list", *[str(x) for x in args.train_steps_list],
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--sigma", str(args.sigma),
        "--eta", str(args.eta),
        "--num_eval_tasks", str(args.num_eval_tasks),
        "--output_dir", str(output_root / "exp_steps_sweep"),
        "--device", args.device,
        "--model_types", *args.model_types,
        "--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios],
        "--lowrank_block_size", str(args.lowrank_block_size),
    ])

    # 2) Layer sweep (existing script)
    run_cmd([
        sys.executable, "scripts/lsa_gd_multilayer.py",
        "--d", str(args.d),
        "--num_layers_list", *[str(x) for x in args.num_layers_list],
        "--num_train_tasks", str(args.num_train_tasks),
        "--num_eval_tasks", str(args.num_eval_tasks),
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--lr", str(args.lr),
        "--eta", str(args.eta),
        "--sigma", str(args.sigma),
        "--output_dir", str(output_root / "exp_layers_sweep"),
        "--device", args.device,
        "--model_types", *args.model_types,
        "--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios],
        "--lowrank_block_size", str(args.lowrank_block_size),
    ])

    # 3) In-context sweep
    run_cmd([
        sys.executable, "scripts/exp_context_sweep.py",
        "--d", str(args.d),
        "--n_points_list", *[str(x) for x in args.n_points_list],
        "--num_layers", str(args.steps_num_layers),
        "--train_steps", str(args.context_train_steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--sigma", str(args.sigma),
        "--eta", str(args.eta),
        "--num_eval_tasks", str(args.num_eval_tasks),
        "--output_dir", str(output_root / "exp_context_sweep"),
        "--device", args.device,
        "--model_types", *args.model_types,
        "--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios],
        "--lowrank_block_size", str(args.lowrank_block_size),
    ])


if __name__ == '__main__':
    main()
