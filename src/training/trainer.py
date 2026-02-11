"""Training loop for Q2 attention mechanism experiments."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import save_config
from src.data.task_sampler import LinearRegressionTaskSampler
from src.evaluation.metrics import evaluate_batches
from src.models import build_model_from_config
from src.utils.io import append_jsonl, ensure_dir, write_json
from src.utils.seed import set_seed


class Trainer:
    """Trainer for in-context linear regression experiments."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config["seed"])

        self.output_dir = ensure_dir(config["output_dir"])
        self.ckpt_dir = ensure_dir(self.output_dir / "checkpoints")
        self.log_path = self.output_dir / "train_log.jsonl"

        save_config(config, self.output_dir / "config_resolved.json")

        self.model = build_model_from_config(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"]["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_schedule,
        )
        self.criterion = nn.MSELoss()

        self.sampler = LinearRegressionTaskSampler(
            task_dim=config["model"]["task_dim"],
            device=self.device,
        )

        self.start_step = 0
        if config["train"].get("resume", True):
            self._load_latest_checkpoint()

    def _lr_schedule(self, step: int) -> float:
        warmup = max(self.config["train"]["warmup_steps"], 1)
        total = max(self.config["train"]["total_steps"], warmup + 1)
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / (total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _curriculum_dim(self, step: int) -> int:
        cur = self.config["train"]["curriculum"]
        if not cur["enabled"]:
            return self.config["model"]["task_dim"]
        if step >= cur["end_step"]:
            return cur["end_dim"]
        alpha = step / max(cur["end_step"], 1)
        value = cur["start_dim"] + alpha * (cur["end_dim"] - cur["start_dim"])
        return int(round(value))

    def _curriculum_context(self, step: int) -> int:
        cur = self.config["train"]["curriculum"]
        if not cur["enabled"]:
            return self.config["model"]["max_context"]
        if step >= cur["context_end_step"]:
            return cur["end_context"]
        alpha = step / max(cur["context_end_step"], 1)
        value = cur["start_context"] + alpha * (cur["end_context"] - cur["start_context"])
        return int(round(value))

    def _checkpoint_path(self, step: Optional[int] = None) -> Path:
        if step is None:
            return self.ckpt_dir / "checkpoint_latest.pt"
        return self.ckpt_dir / f"checkpoint_{step}.pt"

    def _save_checkpoint(self, step: int, best_eval_mse: float) -> None:
        payload = {
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_eval_mse": best_eval_mse,
            "config": self.config,
        }
        torch.save(payload, self._checkpoint_path())

        interval = self.config["train"]["checkpoint_interval"]
        if step % interval == 0:
            torch.save(payload, self._checkpoint_path(step))

    def _load_latest_checkpoint(self) -> None:
        path = self._checkpoint_path()
        if not path.exists():
            return

        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.start_step = int(payload.get("step", 0))

    def _run_eval(self, step: int, n_context: int, active_dim: int) -> Dict[str, float]:
        eval_cfg = self.config["eval"]
        data_cfg = self.config["data"]
        metrics = evaluate_batches(
            model=self.model,
            sampler=self.sampler,
            num_prompts=eval_cfg["num_prompts"],
            batch_size=eval_cfg["batch_size"],
            task_dim=self.config["model"]["task_dim"],
            n_context=n_context,
            active_dim=active_dim,
            noise_std=data_cfg["noise_std"],
            x_distribution=data_cfg["x_distribution"],
            skew_power=data_cfg["skew_power"],
            orthant_mode=data_cfg["orthant_mode"],
            gd_eta=eval_cfg["gd_eta"],
            compute_baselines=eval_cfg["compute_baselines"],
            compute_alignment=eval_cfg["compute_alignment"],
            knn_k=eval_cfg["knn_k"],
            ridge_lambda=eval_cfg["ridge_lambda"],
            device=self.device,
        )

        record = {
            "kind": "eval",
            "step": step,
            "n_context": n_context,
            "active_dim": active_dim,
            **metrics,
        }
        append_jsonl(self.log_path, record)
        return metrics

    def train(self) -> Dict[str, float]:
        train_cfg = self.config["train"]
        data_cfg = self.config["data"]
        loss_mode = train_cfg.get("loss_mode", "sequence")
        if loss_mode not in {"sequence", "query"}:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

        best_eval_mse = float("inf")
        best_metrics: Dict[str, float] = {}

        pbar = tqdm(range(self.start_step, train_cfg["total_steps"]), desc="train")
        for step in pbar:
            self.model.train()

            active_dim = self._curriculum_dim(step)
            n_context = self._curriculum_context(step)

            batch = self.sampler.sample_batch(
                batch_size=train_cfg["batch_size"],
                n_context=n_context,
                active_dim=active_dim,
                noise_std=data_cfg["noise_std"],
                x_distribution=data_cfg["x_distribution"],
                skew_power=data_cfg["skew_power"],
                orthant_mode=data_cfg["orthant_mode"],
            )

            x_context = batch["x_context"]
            y_context = batch["y_context"]
            x_query = batch["x_query"]
            y_query = batch["y_query"].squeeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            if loss_mode == "sequence":
                xs = torch.cat([x_context, x_query], dim=1)
                ys = torch.cat([y_context, y_query.unsqueeze(1)], dim=1)
                y_pred = self.model.predict_sequence(xs, ys)
                loss = self.criterion(y_pred, ys)
            else:
                y_pred = self.model.predict(x_context, y_context, x_query)
                loss = self.criterion(y_pred, y_query)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg["grad_clip"])
            self.optimizer.step()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", d=active_dim, n=n_context, lr=f"{lr:.2e}")

            if step % train_cfg["log_interval"] == 0:
                append_jsonl(
                    self.log_path,
                    {
                        "kind": "train",
                        "step": step,
                        "loss": float(loss.item()),
                        "lr": lr,
                        "active_dim": active_dim,
                        "n_context": n_context,
                        "loss_mode": loss_mode,
                    },
                )

            if step % train_cfg["eval_interval"] == 0 and step > 0:
                metrics = self._run_eval(step=step, n_context=n_context, active_dim=active_dim)
                if metrics["mse"] < best_eval_mse:
                    best_eval_mse = metrics["mse"]
                    best_metrics = metrics
                    torch.save(
                        {
                            "step": step,
                            "model_state": self.model.state_dict(),
                            "metrics": metrics,
                            "config": self.config,
                        },
                        self.ckpt_dir / "checkpoint_best.pt",
                    )

            if step % train_cfg["checkpoint_interval"] == 0 and step > 0:
                self._save_checkpoint(step=step, best_eval_mse=best_eval_mse)

        self._save_checkpoint(step=train_cfg["total_steps"], best_eval_mse=best_eval_mse)
        if best_metrics:
            write_json(self.output_dir / "best_eval_metrics.json", best_metrics)
        return best_metrics
