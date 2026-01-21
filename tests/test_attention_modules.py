import torch
import sys
sys.path.insert(0, '..')

from models.attention.baseline_attention import BaselineAttention
from models.attention.linear_attention import LinearAttention
from models.attention.lowrank_attention import LowRankAttention


def test_attention_shapes():
    """Test that all attention modules produce correct output shapes."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model)
    
    # Test baseline
    baseline = BaselineAttention(d_model, n_heads=8)
    output = baseline(x)
    assert output.shape == (batch, n, d_model), f"Baseline: Expected {(batch, n, d_model)}, got {output.shape}"
    
    # Test linear
    linear = LinearAttention(d_model, n_heads=8)
    output = linear(x)
    assert output.shape == (batch, n, d_model), f"Linear: Expected {(batch, n, d_model)}, got {output.shape}"
    
    # Test lowrank
    lowrank = LowRankAttention(d_model, n_heads=8, rank=16)
    output = lowrank(x)
    assert output.shape == (batch, n, d_model), f"LowRank: Expected {(batch, n, d_model)}, got {output.shape}"
    
    print("✓ Shape tests passed")


def test_no_nans():
    """Test that all attention modules produce stable outputs."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model)
    
    modules = {
        'baseline': BaselineAttention(d_model, n_heads=8),
        'linear': LinearAttention(d_model, n_heads=8),
        'lowrank': LowRankAttention(d_model, n_heads=8, rank=16)
    }
    
    for name, module in modules.items():
        output = module(x)
        assert not torch.isnan(output).any(), f"{name}: NaN detected"
        assert not torch.isinf(output).any(), f"{name}: Inf detected"
    
    print("✓ NaN/Inf tests passed")


def test_gradients():
    """Test gradient flow through all modules."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model, requires_grad=True)
    
    modules = {
        'baseline': BaselineAttention(d_model, n_heads=8),
        'linear': LinearAttention(d_model, n_heads=8),
        'lowrank': LowRankAttention(d_model, n_heads=8, rank=16)
    }
    
    for name, module in modules.items():
        x.grad = None  # Reset
        output = module(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, f"{name}: No gradients for input"
        assert not torch.isnan(x.grad).any(), f"{name}: NaN gradients"
    
    print("✓ Gradient tests passed")


if __name__ == '__main__':
    print("Running attention module tests...")
    test_attention_shapes()
    test_no_nans()
    test_gradients()
    print("\n✓ All tests passed!")
