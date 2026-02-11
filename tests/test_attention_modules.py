import torch

from src.models.attention.baseline_attention import BaselineAttention
from src.models.attention.linear_attention import LinearAttention
from src.models.attention.lowrank_attention import LowRankAttention


def _run_attention(attn):
    x = torch.randn(2, 11, 64)
    mask = torch.tril(torch.ones(11, 11, dtype=torch.bool))
    if isinstance(attn, LowRankAttention):
        out, weights = attn(x, mask=None, return_attn=True)
    else:
        out, weights = attn(x, mask=mask, return_attn=True)
    assert out.shape == (2, 11, 64)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert weights is not None


def test_baseline_attention_shapes():
    _run_attention(BaselineAttention(d_model=64, n_heads=4))


def test_linear_attention_shapes():
    _run_attention(LinearAttention(d_model=64, n_heads=4))


def test_lowrank_attention_shapes():
    _run_attention(LowRankAttention(d_model=64, n_heads=4, rank=8, max_seq_len=32))
