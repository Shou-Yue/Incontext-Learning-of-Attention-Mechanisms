#!/usr/bin/env python3
"""
Master runner for all 4 experiments.

Creates timestamped output under results/experiments_<time>_<date>/.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_cmd(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


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


def _merge_results(existing, new, key):
    merged = {int(r.get(key)): r for r in existing if key in r}
    for r in new:
        if key in r:
            k = int(r[key])
            if k in merged:
                merged_entry = dict(merged[k])
                merged_entry.update(r)
                merged[k] = merged_entry
            else:
                merged[k] = r
    return [merged[k] for k in sorted(merged.keys())]


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
        # pattern: <model_name>_<N>layer_checkpoint.pt
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


def _split_missing_models(missing_models):
    base_models = []
    lowrank_ratios = []
    for m in missing_models:
        if m.startswith("lowrank_"):
            # m like lowrank_block2_k0.5 or lowrank_k0.5
            try:
                ratio_str = m.split("_k")[-1]
                lowrank_ratios.append(float(ratio_str))
            except Exception:
                continue
        else:
            base_models.append(m)
    return sorted(set(base_models)), sorted(set(lowrank_ratios))


def _collect_completed_layers(exp_dir: Path, expected_models):
    completed = set()
    # If all_results exists, assume layers present there are complete
    for r in _load_results(exp_dir / "all_results.json"):
        if "num_layers" in r:
            completed.add(int(r["num_layers"]))
    if completed:
        return completed
    # Otherwise use checkpoint presence
    missing_map = _layer_missing_map(exp_dir, expected_models)
    for layer, missing in missing_map.items():
        if not missing:
            completed.add(layer)
    return completed


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


def _load_existing_steps_results(exp_dir: Path):
    results = _load_results(exp_dir / "all_results.json")
    if results:
        return results
    # Fallback to per-step files
    out = []
    for f in sorted(exp_dir.glob("results_steps_*.json")):
        try:
            data = json.loads(f.read_text())
            if "train_steps" in data:
                out.append(data)
        except Exception:
            continue
    return out


def _load_existing_context_results(exp_dir: Path):
    results = _load_results(exp_dir / "all_results.json")
    if results:
        return results
    out = []
    for f in sorted(exp_dir.glob("results_n_*.json")):
        try:
            data = json.loads(f.read_text())
            if "n_points" in data:
                out.append(data)
        except Exception:
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description="Run all four experiments")
    parser.add_argument('--output_root', type=str, default='results',
                        help='Root results directory')
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from existing experiments_<timestamp> folder')
    parser.add_argument('--no_bump_timestamp', action='store_true',
                        help='Do not rename the resume folder with a new timestamp after completion')

    # Shared model args
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['lsa', 'softmax', 'linformer', 'kernel', 'gla', 'gqa', 'sparse'])
    parser.add_argument('--lowrank_k_ratios', type=float, nargs='+',
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--lowrank_block_size', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision (AMP) on CUDA')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile on models (PyTorch 2.x)')

    # Steps sweep
    parser.add_argument('--steps_n_points', type=int, default=41)
    parser.add_argument('--steps_num_layers', type=int, default=8)
    parser.add_argument('--train_steps_list', type=int, nargs='+',
                        default=[0, 1000, 2000, 5000, 10000, 20000])

    # Layer sweep
    parser.add_argument('--num_layers_list', type=int, nargs='+',
                        default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--num_train_tasks', type=int, default=64000)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_eval_tasks', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.0)

    # Context sweep (trained)
    parser.add_argument('--n_points_list', type=int, nargs='+',
                        default=[5, 10, 20, 40])
    parser.add_argument('--context_train_steps', type=int, default=10000)

    # Context sweep (zero train)
    parser.add_argument('--n_points_zero_list', type=int, nargs='+',
                        default=[5, 10, 20, 40, 80])
    parser.add_argument('--zero_num_layers', type=int, default=8)

    args = parser.parse_args()

    resume_mode = args.resume_from is not None
    if resume_mode:
        exp_dir = Path(args.resume_from).expanduser()
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
        exp_dir = Path(args.output_root) / f"experiments_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Training steps sweep
    steps_dir = exp_dir / "exp_steps_sweep"
    if resume_mode:
        completed = _collect_completed_steps(steps_dir)
        old_results = _load_existing_steps_results(steps_dir)
        missing_steps = [s for s in args.train_steps_list if s not in completed]
    else:
        missing_steps = list(args.train_steps_list)
        completed = []
        old_results = []

    if missing_steps:
        cmd = [
            sys.executable, "scripts/exp_steps_sweep.py",
            "--d", str(args.d),
            "--n_points", str(args.steps_n_points),
            "--num_layers", str(args.steps_num_layers),
            "--train_steps_list", *[str(x) for x in missing_steps],
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--sigma", str(args.sigma),
            "--eta", str(args.eta),
            "--num_eval_tasks", str(args.num_eval_tasks),
            "--output_dir", str(steps_dir),
            "--device", args.device,
            "--model_types", *args.model_types,
            "--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios],
            "--lowrank_block_size", str(args.lowrank_block_size),
        ]
        if args.use_amp:
            cmd.append("--use_amp")
        if args.compile:
            cmd.append("--compile")
        run_cmd(cmd)
        if resume_mode and old_results:
            new_results = _load_results(steps_dir / "all_results.json")
            merged = _merge_results(old_results, new_results, "train_steps")
            (steps_dir / "all_results.json").write_text(json.dumps(merged, indent=2))
    else:
        print("\n[skip] steps sweep already complete")

    # 2) Layer sweep (trained)
    layers_dir = exp_dir / "exp_layers_sweep"
    expected_models = _expected_layer_models(args)
    if resume_mode:
        completed = _collect_completed_layers(layers_dir, expected_models)
        missing_layers = [l for l in args.num_layers_list if l not in completed]
    else:
        missing_layers = list(args.num_layers_list)
        completed = []

    if missing_layers:
        old_results = _load_results(layers_dir / "all_results.json") if resume_mode else []
        if resume_mode:
            missing_detail = _layer_missing_map(layers_dir, expected_models)
            for layer in missing_layers:
                missing_models = missing_detail.get(layer, expected_models)
                if not missing_models:
                    continue

                base_models, lowrank_ratios = _split_missing_models(missing_models)
                model_types = list(base_models)
                cmd = [
                    sys.executable, "scripts/lsa_gd_multilayer.py",
                    "--d", str(args.d),
                    "--num_layers_list", str(layer),
                    "--num_train_tasks", str(args.num_train_tasks),
                    "--num_eval_tasks", str(args.num_eval_tasks),
                    "--batch_size", str(args.batch_size),
                    "--num_epochs", str(args.num_epochs),
                    "--lr", str(args.lr),
                    "--eta", str(args.eta),
                    "--sigma", str(args.sigma),
                    "--output_dir", str(layers_dir),
                    "--device", args.device,
                ]

                if lowrank_ratios:
                    model_types.append("linformer")
                if not model_types:
                    continue

                cmd += ["--model_types", *model_types]
                if lowrank_ratios:
                    cmd += ["--lowrank_k_ratios", *[str(x) for x in lowrank_ratios]]
                else:
                    cmd += ["--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios]]
                cmd += ["--lowrank_block_size", str(args.lowrank_block_size)]
                if args.use_amp:
                    cmd.append("--use_amp")
                if args.compile:
                    cmd.append("--compile")
                run_cmd(cmd)
        else:
            cmd = [
                sys.executable, "scripts/lsa_gd_multilayer.py",
                "--d", str(args.d),
                "--num_layers_list", *[str(x) for x in missing_layers],
                "--num_train_tasks", str(args.num_train_tasks),
                "--num_eval_tasks", str(args.num_eval_tasks),
                "--batch_size", str(args.batch_size),
                "--num_epochs", str(args.num_epochs),
                "--lr", str(args.lr),
                "--eta", str(args.eta),
                "--sigma", str(args.sigma),
                "--output_dir", str(layers_dir),
                "--device", args.device,
                "--model_types", *args.model_types,
                "--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios],
                "--lowrank_block_size", str(args.lowrank_block_size),
            ]
            if args.use_amp:
                cmd.append("--use_amp")
            if args.compile:
                cmd.append("--compile")
            run_cmd(cmd)
        if resume_mode and old_results:
            new_results = _load_results(layers_dir / "all_results.json")
            merged = _merge_results(old_results, new_results, "num_layers")
            (layers_dir / "all_results.json").write_text(json.dumps(merged, indent=2))
    else:
        print("\n[skip] layers sweep already complete")

    # 3) In-context sweep (trained)
    context_dir = exp_dir / "exp_context_sweep"
    if resume_mode:
        completed = _collect_completed_context(context_dir)
        missing_ctx = [n for n in args.n_points_list if n not in completed]
        old_results = _load_existing_context_results(context_dir)
    else:
        missing_ctx = list(args.n_points_list)
        completed = []
        old_results = []

    if missing_ctx:
        old_results = old_results if resume_mode else []
        cmd = [
            sys.executable, "scripts/exp_context_sweep.py",
            "--d", str(args.d),
            "--n_points_list", *[str(x) for x in missing_ctx],
            "--num_layers", str(args.steps_num_layers),
            "--train_steps", str(args.context_train_steps),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--sigma", str(args.sigma),
            "--eta", str(args.eta),
            "--num_eval_tasks", str(args.num_eval_tasks),
            "--output_dir", str(context_dir),
            "--device", args.device,
            "--model_types", *args.model_types,
            "--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios],
            "--lowrank_block_size", str(args.lowrank_block_size),
        ]
        if args.use_amp:
            cmd.append("--use_amp")
        if args.compile:
            cmd.append("--compile")
        run_cmd(cmd)
        if resume_mode and old_results:
            new_results = _load_results(context_dir / "all_results.json")
            merged = _merge_results(old_results, new_results, "n_points")
            (context_dir / "all_results.json").write_text(json.dumps(merged, indent=2))
    else:
        print("\n[skip] context sweep already complete")

    # 4) In-context sweep (zero training)
    context_zero_dir = exp_dir / "exp_context_sweep_zero"
    if resume_mode:
        completed = _collect_completed_context(context_zero_dir)
        missing_zero = [n for n in args.n_points_zero_list if n not in completed]
        old_results = _load_existing_context_results(context_zero_dir)
    else:
        missing_zero = list(args.n_points_zero_list)
        completed = []
        old_results = []

    if missing_zero:
        old_results = old_results if resume_mode else []
        run_cmd([
            sys.executable, "scripts/exp_context_sweep_zero_train.py",
            "--d", str(args.d),
            "--n_points_list", *[str(x) for x in missing_zero],
            "--num_layers", str(args.zero_num_layers),
            "--num_eval_tasks", str(args.num_eval_tasks),
            "--eta", str(args.eta),
            "--sigma", str(args.sigma),
            "--output_dir", str(context_zero_dir),
            "--device", args.device,
            "--model_types", *args.model_types,
            "--lowrank_k_ratios", *[str(x) for x in args.lowrank_k_ratios],
            "--lowrank_block_size", str(args.lowrank_block_size),
        ])
        if resume_mode and old_results:
            new_results = _load_results(context_zero_dir / "all_results.json")
            merged = _merge_results(old_results, new_results, "n_points")
            (context_zero_dir / "all_results.json").write_text(json.dumps(merged, indent=2))
    else:
        print("\n[skip] zero-train context sweep already complete")

    # Plot all 8 graphs
    run_cmd([
        sys.executable, "scripts/plot_all_experiments.py",
        "--steps_results", str(steps_dir / "all_results.json"),
        "--layers_results", str(layers_dir / "all_results.json"),
        "--context_results", str(context_dir / "all_results.json"),
        "--context_zero_results", str(context_zero_dir / "all_results.json"),
        "--output_dir", str(plots_dir),
    ])

    final_dir = exp_dir
    if resume_mode and not args.no_bump_timestamp:
        new_stamp = datetime.now().strftime("%H%M%S_%Y%m%d")
        bumped = exp_dir.parent / f"experiments_{new_stamp}"
        if bumped != exp_dir and not bumped.exists():
            exp_dir.rename(bumped)
            final_dir = bumped
        else:
            print(f"[warn] Could not bump timestamp; folder exists: {bumped}")

    print(f"\nAll results in: {final_dir}")


if __name__ == '__main__':
    main()
