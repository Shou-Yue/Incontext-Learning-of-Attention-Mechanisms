#!/usr/bin/env python3
"""
Sanity check: low-rank (k=1.0, identity) should match softmax exactly
when weights are copied.
"""
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.models.attention_variants import MultiLayerAttentionModel


def main():
    torch.manual_seed(0)
    d = 20
    num_layers = 2
    n_points = 2 * d + 1
    n_tokens = n_points + 1
    batch_size = 4

    softmax = MultiLayerAttentionModel(
        d=d,
        num_layers=num_layers,
        hidden_dim=None,
        attn_type='softmax',
        causal=False
    )
    lowrank = MultiLayerAttentionModel(
        d=d,
        num_layers=num_layers,
        hidden_dim=None,
        attn_type='linformer',
        n_tokens=n_tokens,
        proj_k=n_tokens,
        lowrank_kwargs={
            'identity_proj': True,
            'freeze_proj': True,
            'normalize_qk': False,
            'learnable_scale': False
        }
    )

    # Copy weights from softmax to lowrank (excluding E/F which are identity)
    soft_state = softmax.state_dict()
    low_state = lowrank.state_dict()
    for k, v in soft_state.items():
        if k in low_state and low_state[k].shape == v.shape:
            low_state[k].copy_(v)
    lowrank.load_state_dict(low_state)

    xs = torch.randn(batch_size, n_points, d)
    ys = torch.randn(batch_size, n_points)
    query_x = torch.randn(batch_size, 1, d)

    with torch.no_grad():
        y_soft = softmax(xs, ys, query_x)
        y_low = lowrank(xs, ys, query_x)

    diff = (y_soft - y_low).abs().max().item()
    print(f"Max abs diff: {diff:.8f}")


if __name__ == "__main__":
    main()
