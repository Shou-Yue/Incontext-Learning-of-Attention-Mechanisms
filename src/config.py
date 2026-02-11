"""Configuration helpers for Q2 attention mechanism experiments."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_name": "q2_default",
    "seed": 42,
    "output_dir": "results/default",
    "model": {
        "attn_type": "baseline",  # baseline | linear | lowrank
        "attn_rank": 16,
        "attn_feature_map": "elu",
        "task_dim": 20,
        "max_context": 40,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "d_mlp": 1024,
        "dropout": 0.0,
    },
    "data": {
        "noise_std": 0.0,
        "x_distribution": "isotropic",  # isotropic | skewed
        "skew_power": 2.0,
        "orthant_mode": "none",  # none | split
    },
    "train": {
        "batch_size": 64,
        "total_steps": 100000,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "warmup_steps": 5000,
        "grad_clip": 1.0,
        "loss_mode": "sequence",  # sequence | query
        "log_interval": 100,
        "eval_interval": 1000,
        "checkpoint_interval": 5000,
        "resume": True,
        "curriculum": {
            "enabled": True,
            "start_dim": 5,
            "end_dim": 20,
            "end_step": 20000,
            "start_context": 5,
            "end_context": 40,
            "context_end_step": 20000,
        },
    },
    "eval": {
        "num_prompts": 1280,
        "batch_size": 128,
        "gd_eta": 1.0,
        "knn_k": 3,
        "ridge_lambda": 1e-2,
        "compute_baselines": True,
        "compute_alignment": True,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a config dictionary merged with default values."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    return _deep_update(merged, config)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load config JSON and merge with defaults."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    merged = with_defaults(config)
    merged["_config_path"] = str(path)
    return merged


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """Save config dictionary to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


__all__ = ["DEFAULT_CONFIG", "load_config", "save_config", "with_defaults"]
