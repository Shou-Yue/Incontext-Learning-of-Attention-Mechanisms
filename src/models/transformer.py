"""Transformer backbone with swappable attention mechanisms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.data.task_sampler import build_prompt_tokens
from src.models.attention import BaselineAttention, LinearAttention, LowRankAttention


@dataclass
class ModelSpec:
    attn_type: str
    task_dim: int
    max_context: int
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    dropout: float = 0.0
    attn_rank: int = 16
    attn_feature_map: str = "elu"

    @property
    def max_seq_len(self) -> int:
        # +1 point supports sequence-wide training on n_context + query.
        return 2 * (self.max_context + 1)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with interchangeable attention."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(spec.d_model)
        self.ln2 = nn.LayerNorm(spec.d_model)

        if spec.attn_type == "baseline":
            self.attn = BaselineAttention(spec.d_model, spec.n_heads, dropout=spec.dropout)
        elif spec.attn_type == "linear":
            self.attn = LinearAttention(spec.d_model, spec.n_heads, feature_map=spec.attn_feature_map)
        elif spec.attn_type == "lowrank":
            self.attn = LowRankAttention(
                spec.d_model,
                spec.n_heads,
                rank=spec.attn_rank,
                max_seq_len=spec.max_seq_len,
            )
        else:
            raise ValueError(f"Unknown attn_type: {spec.attn_type}")

        self.mlp = nn.Sequential(
            nn.Linear(spec.d_model, spec.d_mlp),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_mlp, spec.d_model),
            nn.Dropout(spec.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_mask = None if isinstance(self.attn, LowRankAttention) else mask
        attn_out, attn = self.attn(self.ln1(x), mask=attn_mask, return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn


class InContextTransformer(nn.Module):
    """Model for in-context linear regression prompts."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec

        max_seq_len = spec.max_seq_len
        self.read_in = nn.Linear(spec.task_dim, spec.d_model)
        self.pos_emb = nn.Embedding(max_seq_len, spec.d_model)
        self.dropout = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(spec) for _ in range(spec.n_layers)])
        self.final_ln = nn.LayerNorm(spec.d_model)
        self.read_out = nn.Linear(spec.d_model, 1)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def _run_backbone(
        self,
        tokens: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        bsz, seq_len, _ = tokens.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"Input seq_len={seq_len} exceeds positional capacity={self.pos_emb.num_embeddings}"
            )

        positions = torch.arange(seq_len, device=tokens.device)
        h = self.read_in(tokens) + self.pos_emb(positions).view(1, seq_len, -1)
        h = self.dropout(h)
        mask = self._causal_mask(seq_len, tokens.device)

        all_attn: List[torch.Tensor] = []
        for block in self.blocks:
            h, attn = block(h, mask=mask, return_attn=return_attn)
            if return_attn and attn is not None:
                all_attn.append(attn)

        h = self.final_ln(h)
        if return_attn:
            return h, all_attn
        return h, None

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        tokens = build_prompt_tokens(x_context, y_context, x_query)
        h, all_attn = self._run_backbone(tokens=tokens, return_attn=return_attn)
        y_pred = self.read_out(h[:, -1, :]).squeeze(-1)

        if return_attn:
            return y_pred, all_attn
        return y_pred, None

    def forward_sequence(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Predict all y positions for interleaved sequence training."""
        batch_size, n_points, dim = xs.shape
        ys_wide = torch.zeros(batch_size, n_points, dim, device=xs.device)
        ys_wide[:, :, 0] = ys
        tokens = torch.stack([xs, ys_wide], dim=2).view(batch_size, 2 * n_points, dim)

        h, all_attn = self._run_backbone(tokens=tokens, return_attn=return_attn)
        all_scalar = self.read_out(h).squeeze(-1)
        y_pred = all_scalar[:, 1::2]
        if return_attn:
            return y_pred, all_attn
        return y_pred, None

    def predict(self, x_context: torch.Tensor, y_context: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        y_pred, _ = self.forward(x_context, y_context, x_query, return_attn=False)
        return y_pred

    def predict_sequence(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        y_pred, _ = self.forward_sequence(xs, ys, return_attn=False)
        return y_pred

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(config: Dict[str, Dict]) -> InContextTransformer:
    model_cfg = config["model"]
    spec = ModelSpec(
        attn_type=model_cfg["attn_type"],
        task_dim=model_cfg["task_dim"],
        max_context=model_cfg["max_context"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_mlp=model_cfg["d_mlp"],
        dropout=model_cfg["dropout"],
        attn_rank=model_cfg.get("attn_rank", 16),
        attn_feature_map=model_cfg.get("attn_feature_map", "elu"),
    )
    return InContextTransformer(spec)


__all__ = ["ModelSpec", "InContextTransformer", "build_model_from_config"]
