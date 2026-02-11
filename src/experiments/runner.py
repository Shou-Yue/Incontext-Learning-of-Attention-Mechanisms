"""Shared helpers for Q2 experiment scripts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch

from src.config import with_defaults
from src.data.task_sampler import LinearRegressionTaskSampler
from src.evaluation.metrics import evaluate_batches
from src.models import build_model_from_config
from src.training.trainer import Trainer
from src.utils.io import ensure_dir, write_json


def merged_config(base: Dict, overrides: Dict) -> Dict:
    cfg = deepcopy(base)

    def _merge(a: Dict, b: Dict) -> Dict:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                _merge(a[k], v)
            else:
                a[k] = v
        return a

    return with_defaults(_merge(cfg, overrides))


def train_and_get_checkpoint(config: Dict) -> Path:
    trainer = Trainer(config)
    trainer.train()
    best = Path(config["output_dir"]) / "checkpoints" / "checkpoint_best.pt"
    if best.exists():
        return best
    return Path(config["output_dir"]) / "checkpoints" / "checkpoint_latest.pt"


def load_model_from_checkpoint(config: Dict, checkpoint_path: str | Path):
    model = build_model_from_config(config)
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    return model


def evaluate_condition(config: Dict, checkpoint_path: str | Path, n_context: int, active_dim: int, data_overrides: Dict | None = None) -> Dict[str, float]:
    model = load_model_from_checkpoint(config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_cfg = deepcopy(config["data"])
    if data_overrides:
        data_cfg.update(data_overrides)

    sampler = LinearRegressionTaskSampler(task_dim=config["model"]["task_dim"], device=device)
    return evaluate_batches(
        model=model,
        sampler=sampler,
        num_prompts=config["eval"]["num_prompts"],
        batch_size=config["eval"]["batch_size"],
        task_dim=config["model"]["task_dim"],
        n_context=n_context,
        active_dim=active_dim,
        noise_std=data_cfg["noise_std"],
        x_distribution=data_cfg["x_distribution"],
        skew_power=data_cfg["skew_power"],
        orthant_mode=data_cfg["orthant_mode"],
        gd_eta=config["eval"]["gd_eta"],
        compute_baselines=config["eval"]["compute_baselines"],
        compute_alignment=config["eval"]["compute_alignment"],
        knn_k=config["eval"]["knn_k"],
        ridge_lambda=config["eval"]["ridge_lambda"],
        device=device,
    )


def save_experiment_result(output_dir: str | Path, filename: str, payload: Dict) -> None:
    out_dir = ensure_dir(output_dir)
    write_json(out_dir / filename, payload)

