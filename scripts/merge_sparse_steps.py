#!/usr/bin/env python3
"""
Merge ONLY sparse steps-sweep results into the main exp_steps_sweep folder.
Preserves all other attention mechanisms.
"""
import argparse
import json
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _merge(existing, new):
    merged = {r["train_steps"]: dict(r) for r in existing if "train_steps" in r}
    for r in new:
        if "train_steps" not in r:
            continue
        key = r["train_steps"]
        if key in merged:
            merged[key].update(r)
        else:
            merged[key] = dict(r)
    return [merged[k] for k in sorted(merged.keys())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to results/experiments_* folder.")
    args = parser.parse_args()

    base = Path(args.exp_dir)
    steps_tmp = base / "exp_steps_sweep_sparse_tmp"
    steps_target = base / "exp_steps_sweep"

    tmp_all = steps_tmp / "all_results.json"
    if not tmp_all.exists():
        raise SystemExit(f"Missing {tmp_all}")

    steps_target.mkdir(parents=True, exist_ok=True)
    target_all = steps_target / "all_results.json"

    new = _load(tmp_all)
    existing = _load(target_all)
    merged = _merge(existing, new)
    target_all.write_text(json.dumps(merged, indent=2))

    # Copy per-step files if missing
    for f in steps_tmp.glob("results_steps_*.json"):
        dest = steps_target / f.name
        if not dest.exists():
            dest.write_bytes(f.read_bytes())

    print(f"Merged sparse steps into: {target_all}")


if __name__ == "__main__":
    main()
