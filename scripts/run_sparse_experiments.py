#!/usr/bin/env python3
"""
Run the 4 core experiments for ONLY Sparse Causal attention,
then merge results into existing exp_* folders so plotting keeps working.

Experiments:
  - steps sweep (trained)
  - layers sweep (trained, per-layer step schedule)
  - context sweep (trained)
  - context sweep (zero-train, long n_points)
"""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_cmd(cmd):
    print("\n>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_results(path: Path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def _merge_results(existing, new, key_field):
    merged = {r[key_field]: dict(r) for r in existing if key_field in r}
    for r in new:
        if key_field not in r:
            continue
        key = r[key_field]
        if key in merged:
            merged[key].update(r)
        else:
            merged[key] = dict(r)
    return [merged[k] for k in sorted(merged.keys())]


def _merge_all_results(target_dir: Path, tmp_dir: Path, key_field: str):
    target_path = target_dir / "all_results.json"
    tmp_path = tmp_dir / "all_results.json"
    new_results = _load_results(tmp_path)
    existing = _load_results(target_path)
    merged = _merge_results(existing, new_results, key_field=key_field)
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(merged, f, indent=2)


def _copy_checkpoints(tmp_dir: Path, target_dir: Path):
    if not tmp_dir.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in tmp_dir.glob("*_checkpoint.pt"):
        dest = target_dir / ckpt.name
        if not dest.exists():
            dest.write_bytes(ckpt.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Existing results/experiments_* folder to merge into.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    if args.resume_from is None:
        stamp = datetime.now().strftime("%H%M%S_%Y%m%d")
        base_dir = Path("results") / f"experiments_{stamp}"
    else:
        base_dir = Path(args.resume_from)
    base_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--model_types", "sparse",
    ]
    if args.use_amp:
        common.append("--use_amp")

    # ---- Steps sweep ----
    steps_tmp = base_dir / "exp_steps_sweep_sparse_tmp"
    steps_target = base_dir / "exp_steps_sweep"
    steps_tmp.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "python3", "scripts/exp_steps_sweep.py",
        "--d", "20",
        "--n_points", "41",
        "--num_layers", "8",
        "--train_steps_list", "0", "1000", "2000", "5000", "10000", "20000",
        "--lr", "0.001",
        "--sigma", "0.0",
        "--eta", "0.1",
        "--num_eval_tasks", "1000",
        "--output_dir", str(steps_tmp),
        *common,
    ])
    _merge_all_results(steps_target, steps_tmp, key_field="train_steps")

    # ---- Layers sweep (trained with per-layer steps) ----
    layers_tmp = base_dir / "exp_layers_sweep_sparse_tmp"
    layers_target = base_dir / "exp_layers_sweep"
    layers_tmp.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "python3", "scripts/exp_layers_sweep_stepsched.py",
        "--d", "20",
        "--n_points", "41",
        "--num_layers_list", "2", "4", "8", "16",
        "--steps_shallow", "5000",
        "--steps_deep", "10000",
        "--deep_layer_threshold", "16",
        "--lr", "0.001",
        "--sigma", "0.0",
        "--eta", "0.1",
        "--num_eval_tasks", "1000",
        "--output_dir", str(layers_tmp),
        *common,
    ])
    _copy_checkpoints(layers_tmp, layers_target)
    _merge_all_results(layers_target, layers_tmp, key_field="num_layers")

    # ---- Context sweep (trained, up to 200) ----
    ctx_tmp = base_dir / "exp_context_sweep_sparse_tmp"
    ctx_target = base_dir / "exp_context_sweep"
    ctx_tmp.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "python3", "scripts/exp_context_sweep.py",
        "--d", "20",
        "--n_points_list", "5", "10", "20", "40", "80", "120", "160", "200",
        "--num_layers", "8",
        "--train_steps", "5000",
        "--lr", "0.001",
        "--sigma", "0.0",
        "--eta", "0.1",
        "--num_eval_tasks", "1000",
        "--output_dir", str(ctx_tmp),
        "--merge_into", str(ctx_target),
        *common,
    ])
    _merge_all_results(ctx_target, ctx_tmp, key_field="n_points")

    # ---- Context sweep (zero-train, long) ----
    ctx0_tmp = base_dir / "exp_context_sweep_zero_sparse_tmp"
    ctx0_target = base_dir / "exp_context_sweep_zero"
    ctx0_tmp.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "python3", "scripts/exp_context_sweep_zero_train_long.py",
        "--d", "20",
        "--n_points_list", "5", "10", "20", "40", "80", "120", "160", "200",
        "--num_layers", "8",
        "--num_eval_tasks", "1000",
        "--sigma", "0.0",
        "--eta", "0.1",
        "--output_dir", str(ctx0_tmp),
        *common,
    ])
    _merge_all_results(ctx0_target, ctx0_tmp, key_field="n_points")

    print(f"\nDone. Results merged into: {base_dir}")


if __name__ == "__main__":
    main()
