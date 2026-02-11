"""Kernelized linear attention (Katharopoulos et al., 2020)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """Linear-time self-attention using feature maps and associative reordering."""

    def __init__(self, d_model: int, n_heads: int, feature_map: str = "elu", eps: float = 1e-6) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.feature_map = feature_map
        self.eps = eps

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 2)  # (b, h, n, d)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.d_model)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu":
            return F.elu(x) + 1.0
        raise ValueError(f"Unsupported feature_map: {self.feature_map}")

    def _causal_linear(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v: (b, h, n, d)
        kv = torch.einsum("bhnd,bhne->bhnde", k, v)
        kv_prefix = torch.cumsum(kv, dim=2)
        k_prefix = torch.cumsum(k, dim=2)

        numer = torch.einsum("bhnd,bhnde->bhne", q, kv_prefix)
        denom = torch.einsum("bhnd,bhnd->bhn", q, k_prefix).unsqueeze(-1)
        return numer / (denom + self.eps)

    def _noncausal_linear(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        numer = torch.einsum("bhnd,bhde->bhne", q, kv)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1)
        return numer / (denom + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._phi(self._split_heads(self.w_q(x)))
        k = self._phi(self._split_heads(self.w_k(x)))
        v = self._split_heads(self.w_v(x))

        # We only use optimized path for standard causal mask or no mask.
        if mask is None:
            out_heads = self._noncausal_linear(q, k, v)
        else:
            is_causal = False
            if mask.dim() == 2:
                is_causal = torch.equal(mask, torch.tril(mask.new_ones(mask.shape, dtype=mask.dtype)).bool())
            elif mask.dim() == 3:
                tril = torch.tril(mask.new_ones(mask.size(-2), mask.size(-1), dtype=mask.dtype)).bool()
                is_causal = torch.all(mask == tril.unsqueeze(0)).item()
            if not is_causal:
                raise ValueError("LinearAttention currently supports causal mask or no mask only")
            out_heads = self._causal_linear(q, k, v)

        out = self.w_o(self._merge_heads(out_heads))

        if return_attn:
            # Attention-like matrix for interpretability (phi(Q)phi(K)^T row-normalized).
            sim = torch.einsum("bhnd,bhmd->bhnm", q, k)
            if mask is not None:
                if mask.dim() == 2:
                    sim = sim.masked_fill(~mask.view(1, 1, mask.size(0), mask.size(1)), 0.0)
                else:
                    sim = sim.masked_fill(~mask.unsqueeze(1), 0.0)
            sim = sim / (sim.sum(dim=-1, keepdim=True) + self.eps)
            return out, sim
        return out, None

