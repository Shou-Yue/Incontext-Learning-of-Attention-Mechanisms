"""Synthetic linear regression task generation for ICL prompts."""
from __future__ import annotations

from typing import Dict

import torch


def _sample_isotropic(batch_size: int, n_points: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, n_points, dim, device=device)


def _sample_skewed(batch_size: int, n_points: int, dim: int, skew_power: float, device: torch.device) -> torch.Tensor:
    # Covariance eigenvalues proportional to 1 / i^p.
    scales = torch.arange(1, dim + 1, device=device, dtype=torch.float32)
    scales = torch.pow(scales, -0.5 * skew_power)
    x = torch.randn(batch_size, n_points, dim, device=device)
    return x * scales.view(1, 1, dim)


def _apply_orthant_split(context_x: torch.Tensor, query_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Context in one random orthant, query in the opposite orthant.
    batch_size, _, dim = context_x.shape
    signs = torch.randint(0, 2, (batch_size, dim), device=context_x.device, dtype=torch.int64)
    signs = torch.where(signs == 0, -torch.ones_like(signs), torch.ones_like(signs)).float()
    context_x = context_x.abs() * signs.unsqueeze(1)
    query_x = query_x.abs() * (-signs).unsqueeze(1)
    return context_x, query_x


class LinearRegressionTaskSampler:
    """Sampler for y = w*^T x + noise tasks with configurable data shifts."""

    def __init__(self, task_dim: int, device: str | torch.device = "cpu") -> None:
        self.task_dim = task_dim
        self.device = torch.device(device)

    def _sample_x(
        self,
        batch_size: int,
        n_points: int,
        active_dim: int,
        x_distribution: str,
        skew_power: float,
    ) -> torch.Tensor:
        if x_distribution == "isotropic":
            x = _sample_isotropic(batch_size, n_points, active_dim, self.device)
        elif x_distribution == "skewed":
            x = _sample_skewed(batch_size, n_points, active_dim, skew_power, self.device)
        else:
            raise ValueError(f"Unknown x_distribution: {x_distribution}")

        if active_dim < self.task_dim:
            pad = torch.zeros(batch_size, n_points, self.task_dim - active_dim, device=self.device)
            x = torch.cat([x, pad], dim=-1)
        return x

    def sample_batch(
        self,
        batch_size: int,
        n_context: int,
        active_dim: int,
        noise_std: float = 0.0,
        x_distribution: str = "isotropic",
        skew_power: float = 2.0,
        orthant_mode: str = "none",
    ) -> Dict[str, torch.Tensor]:
        if active_dim > self.task_dim:
            raise ValueError(f"active_dim={active_dim} exceeds task_dim={self.task_dim}")

        # Sample task-specific true weights in active subspace.
        w_active = torch.randn(batch_size, active_dim, device=self.device)
        if active_dim < self.task_dim:
            w = torch.cat(
                [w_active, torch.zeros(batch_size, self.task_dim - active_dim, device=self.device)],
                dim=-1,
            )
        else:
            w = w_active

        x_context = self._sample_x(batch_size, n_context, active_dim, x_distribution, skew_power)
        x_query = self._sample_x(batch_size, 1, active_dim, x_distribution, skew_power)

        if orthant_mode == "split":
            x_context, x_query = _apply_orthant_split(x_context, x_query)
        elif orthant_mode != "none":
            raise ValueError(f"Unknown orthant_mode: {orthant_mode}")

        y_context = torch.einsum("bnd,bd->bn", x_context, w)
        y_query = torch.einsum("bqd,bd->bq", x_query, w)

        if noise_std > 0:
            y_context = y_context + noise_std * torch.randn_like(y_context)
            y_query = y_query + noise_std * torch.randn_like(y_query)

        return {
            "x_context": x_context,
            "y_context": y_context,
            "x_query": x_query,
            "y_query": y_query,
            "w_star": w,
        }


def build_prompt_tokens(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    """Build interleaved prompt tokens [x1, y1, ..., xn, yn, x_query].

    The y tokens encode scalar y in the first coordinate and zero-pad the rest.
    """
    batch_size, n_context, dim = x_context.shape
    tokens = torch.zeros(batch_size, 2 * n_context + 1, dim, device=x_context.device)

    tokens[:, 0 : 2 * n_context : 2, :] = x_context
    tokens[:, 1 : 2 * n_context : 2, 0] = y_context
    tokens[:, -1, :] = x_query.squeeze(1)
    return tokens


__all__ = ["LinearRegressionTaskSampler", "build_prompt_tokens"]
