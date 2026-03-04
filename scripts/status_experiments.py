#!/usr/bin/env python3
"""
Status report for experiment folders.

Usage:
  python scripts/status_experiments.py --exp_dir results/experiments_<timestamp>
"""
import argparse
import json
from pathlib import Path
from datetime import datetime


def _load_results(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _collect_completed_steps(exp_dir: Path):
    completed = set()
    for r in _load_results(exp_dir / "all_results.json"):
        if "train_steps" in r:
            completed.add(int(r["train_steps"]))
    for f in exp_dir.glob("results_steps_*.json"):
        try:
            data = json.loads(f.read_text())
            if "train_steps" in data:
                completed.add(int(data["train_steps"]))
        except Exception:
            continue
    return completed


def _parse_layer_checkpoints(exp_dir: Path):
    layer_to_models = {}
    for ckpt in exp_dir.glob("*layer_checkpoint.pt"):
        name = ckpt.name
        try:
            token = name.split("_")[-2]  # e.g., "4layer"
            if not token.endswith("layer"):
                continue
            layer = int(token.replace("layer", ""))
            model_name = name[: -(len(token) + len("_checkpoint.pt") + 1)]
            layer_to_models.setdefault(layer, set()).add(model_name)
        except Exception:
            continue
    return layer_to_models


def _expected_layer_models(args):
    expected = []
    for mt in args.model_types:
        if mt in ("linformer", "lowrank"):
            for ratio in args.lowrank_k_ratios:
                if args.lowrank_block_size and args.lowrank_block_size > 0:
                    expected.append(f"lowrank_block{args.lowrank_block_size}_k{ratio:g}")
                else:
                    expected.append(f"lowrank_k{ratio:g}")
        else:
            expected.append(mt)
    return sorted(set(expected))


def _layer_missing_map(exp_dir: Path, expected_models):
    layer_to_models = _parse_layer_checkpoints(exp_dir)
    missing = {}
    for layer, present in layer_to_models.items():
        missing[layer] = sorted([m for m in expected_models if m not in present])
    return missing


def _collect_completed_context(exp_dir: Path):
    completed = set()
    for r in _load_results(exp_dir / "all_results.json"):
        if "n_points" in r:
            completed.add(int(r["n_points"]))
    for f in exp_dir.glob("results_n_*.json"):
        try:
            data = json.loads(f.read_text())
            if "n_points" in data:
                completed.add(int(data["n_points"]))
        except Exception:
            continue
    return completed


def _fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _print_status(title, exp_dir, expected, completed, missing_detail=None):
    print(f"\n=== {title} ===")
    if not exp_dir.exists():
        print(f"Missing folder: {exp_dir}")
        return

    missing = [x for x in expected if x not in completed]
    total = len(expected)
    done = len(completed)
    pct = 100.0 * done / total if total else 0.0

    all_results = exp_dir / "all_results.json"
    if all_results.exists():
        last_ts = all_results.stat().st_mtime
        last_path = all_results
    else:
        last_ts = exp_dir.stat().st_mtime
        last_path = exp_dir

    print(f"Folder: {exp_dir}")
    print(f"Last updated: {_fmt_time(last_ts)} ({last_path.name})")
    print(f"Completed: {done}/{total} ({pct:.1f}%)")
    print(f"Done values: {sorted(completed)}")
    print(f"Missing values: {missing}")
    if missing_detail:
        print("Missing per layer:")
        for layer in sorted(missing_detail.keys()):
            if missing_detail[layer]:
                print(f"  Layer {layer}: {missing_detail[layer]}")


def main():
    parser = argparse.ArgumentParser(description="Status report for experiment runs")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to a results/experiments_<timestamp> folder")

    # Expected ranges (defaults match run_all_experiments.py)
    parser.add_argument("--train_steps_list", type=int, nargs='+',
                        default=[0, 1000, 2000, 5000, 10000, 20000])
    parser.add_argument("--num_layers_list", type=int, nargs='+',
                        default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--n_points_list", type=int, nargs='+',
                        default=[5, 10, 20, 40])
    parser.add_argument("--n_points_zero_list", type=int, nargs='+',
                        default=[5, 10, 20, 40, 80])
    parser.add_argument("--model_types", type=str, nargs='+',
                        default=["lsa", "softmax", "linformer", "kernel", "gla", "gqa", "sparse"])
    parser.add_argument("--lowrank_k_ratios", type=float, nargs='+',
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--lowrank_block_size", type=int, default=2)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).expanduser()

    _print_status(
        "Steps sweep",
        exp_dir / "exp_steps_sweep",
        args.train_steps_list,
        _collect_completed_steps(exp_dir / "exp_steps_sweep")
    )
    expected_models = _expected_layer_models(args)
    layers_dir = exp_dir / "exp_layers_sweep"
    missing_detail = _layer_missing_map(layers_dir, expected_models)
    completed_layers = set()
    for layer in args.num_layers_list:
        if layer in missing_detail and not missing_detail[layer]:
            completed_layers.add(layer)
    _print_status(
        "Layers sweep",
        layers_dir,
        args.num_layers_list,
        completed_layers,
        missing_detail=missing_detail
    )
    _print_status(
        "Context sweep (trained)",
        exp_dir / "exp_context_sweep",
        args.n_points_list,
        _collect_completed_context(exp_dir / "exp_context_sweep")
    )
    _print_status(
        "Context sweep (zero train)",
        exp_dir / "exp_context_sweep_zero",
        args.n_points_zero_list,
        _collect_completed_context(exp_dir / "exp_context_sweep_zero")
    )


if __name__ == "__main__":
    main()
