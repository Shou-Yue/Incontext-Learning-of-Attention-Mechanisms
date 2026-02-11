"""Standard multi-head softmax attention."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineAttention(nn.Module):
    """Full softmax self-attention (O(n^2))."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

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

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)

        if mask is not None:
            # mask: (n, n) or (b, n, n), True means allowed.
            if mask.dim() == 2:
                expanded = mask.view(1, 1, mask.size(0), mask.size(1))
            elif mask.dim() == 3:
                expanded = mask.unsqueeze(1)
            else:
                raise ValueError("mask must have shape (n,n) or (b,n,n)")
            scores = scores.masked_fill(~expanded, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.w_o(self._merge_heads(out))

        if return_attn:
            return out, attn
        return out, None

