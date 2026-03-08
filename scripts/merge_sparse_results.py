#!/usr/bin/env python3
"""
Merge sparse-only tmp results into main exp_* folders.

Looks for these tmp dirs under --exp_dir:
  - exp_steps_sweep_sparse_tmp   -> exp_steps_sweep   (key: train_steps)
  - exp_layers_sweep_sparse_tmp  -> exp_layers_sweep  (key: num_layers)
  - exp_context_sweep_sparse_tmp -> exp_context_sweep (key: n_points)
  - exp_context_sweep_zero_sparse_tmp -> exp_context_sweep_zero (key: n_points)
"""
import argparse
import json
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _merge(existing, new, key_field):
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


def _merge_all_results(tmp_dir: Path, target_dir: Path, key_field: str):
    tmp_path = tmp_dir / "all_results.json"
    if not tmp_path.exists():
        print(f"[skip] missing {tmp_path}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "all_results.json"
    new = _load(tmp_path)
    existing = _load(target_path)
    merged = _merge(existing, new, key_field=key_field)
    target_path.write_text(json.dumps(merged, indent=2))
    print(f"[ok] merged {tmp_path} -> {target_path}")


def _copy_checkpoints(tmp_dir: Path, target_dir: Path):
    if not tmp_dir.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in tmp_dir.glob("*_checkpoint.pt"):
        dest = target_dir / ckpt.name
        if not dest.exists():
            dest.write_bytes(ckpt.read_bytes())


def _copy_partial_results(tmp_dir: Path, target_dir: Path):
    if not tmp_dir.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp_dir.glob("results_*.json"):
        dest = target_dir / f.name
        if not dest.exists():
            dest.write_bytes(f.read_bytes())


def main():
    parser = argparse.ArgumentParser(description="Merge sparse tmp results into main exp_* folders.")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to results/experiments_* folder.")
    args = parser.parse_args()

    base = Path(args.exp_dir)
    if not base.exists():
        raise SystemExit(f"Experiment folder not found: {base}")

    # Steps
    steps_tmp = base / "exp_steps_sweep_sparse_tmp"
    steps_target = base / "exp_steps_sweep"
    _merge_all_results(steps_tmp, steps_target, key_field="train_steps")
    _copy_partial_results(steps_tmp, steps_target)

    # Layers
    layers_tmp = base / "exp_layers_sweep_sparse_tmp"
    layers_target = base / "exp_layers_sweep"
    _merge_all_results(layers_tmp, layers_target, key_field="num_layers")
    _copy_checkpoints(layers_tmp, layers_target)
    _copy_partial_results(layers_tmp, layers_target)

    # Context trained
    ctx_tmp = base / "exp_context_sweep_sparse_tmp"
    ctx_target = base / "exp_context_sweep"
    _merge_all_results(ctx_tmp, ctx_target, key_field="n_points")
    _copy_partial_results(ctx_tmp, ctx_target)

    # Context zero
    ctx0_tmp = base / "exp_context_sweep_zero_sparse_tmp"
    ctx0_target = base / "exp_context_sweep_zero"
    _merge_all_results(ctx0_tmp, ctx0_target, key_field="n_points")
    _copy_partial_results(ctx0_tmp, ctx0_target)

    print("\nDone.")


if __name__ == "__main__":
    main()
