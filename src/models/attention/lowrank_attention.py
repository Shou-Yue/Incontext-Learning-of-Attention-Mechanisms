"""Linformer-style low-rank attention via sequence-length projections.

This module is intentionally strict non-causal (encoder-style), matching Linformer
paper usage.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAttention(nn.Module):
    """Low-rank attention using Linformer-style sequence projections.

    Keys/values are projected from sequence length n -> rank before softmax:
        K_low = E^T K, V_low = F^T V
    """

    def __init__(self, d_model: int, n_heads: int, rank: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if rank <= 0:
            raise ValueError(f"rank must be > 0 but got {rank}")
        self.rank = rank
        self.max_seq_len = max_seq_len

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Sequence projections E, F \in R^(max_seq_len x rank).
        self.seq_proj_k = nn.Parameter(torch.empty(max_seq_len, rank))
        self.seq_proj_v = nn.Parameter(torch.empty(max_seq_len, rank))
        nn.init.xavier_uniform_(self.seq_proj_k)
        nn.init.xavier_uniform_(self.seq_proj_v)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 2)  # (b, h, n, d)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        _, _, seq_len, _ = q.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        proj_k = self.seq_proj_k[:seq_len, :]  # (n, rank)
        proj_v = self.seq_proj_v[:seq_len, :]  # (n, rank)

        # Project along sequence dimension.
        k_low = torch.einsum("bhnd,nr->bhrd", k, proj_k)
        v_low = torch.einsum("bhnd,nr->bhrd", v, proj_v)

        if mask is not None:
            raise ValueError("LowRankAttention is strict non-causal and does not accept masks")

        scores = torch.einsum("bhnd,bhrd->bhnr", q, k_low) / math.sqrt(self.d_head)

        attn_low = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnr,bhrd->bhnd", attn_low, v_low)
        out = self.w_o(self._merge_heads(out))

        if return_attn:
            # Reconstruct an approximate n x n map for visualization/analysis.
            back_proj = torch.relu(proj_k)
            attn_full = torch.einsum("bhnr,mr->bhnm", attn_low, back_proj)
            attn_full = attn_full / (attn_full.sum(dim=-1, keepdim=True) + 1e-8)
            return out, attn_full
        return out, None
