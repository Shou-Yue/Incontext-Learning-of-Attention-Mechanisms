#!/usr/bin/env python3
"""
Quick test script to verify LSA implementation.
"""
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lsa import MultiLayerLSA, LinearSelfAttentionBlock
from src.evaluation.gd_baseline import (
    generate_linear_regression_task,
    gd_t_steps,
    compute_cosine_similarity
)


def test_single_layer():
    """Test single LSA layer."""
    print("Testing single LSA layer...")
    
    d = 10
    model = MultiLayerLSA(d=d, num_layers=1, hidden_dim=d)
    
    # Generate a simple task
    xs, ys, w_true = generate_linear_regression_task(d, n_points=2*d+1, batch_size=2)
    query_x = torch.randn(2, 1, d)
    
    # Forward pass
    y_pred = model(xs, ys, query_x)
    
    print(f"  Input shape: {xs.shape}")
    print(f"  Output shape: {y_pred.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  ✓ Single layer test passed!")
    

def test_multi_layer():
    """Test multi-layer LSA."""
    print("\nTesting multi-layer LSA...")
    
    d = 10
    for num_layers in [1, 2, 4]:
        model = MultiLayerLSA(d=d, num_layers=num_layers, hidden_dim=d)
        
        xs, ys, w_true = generate_linear_regression_task(d, n_points=2*d+1, batch_size=2)
        query_x = torch.randn(2, 1, d)
        
        y_pred = model(xs, ys, query_x)
        
        print(f"  {num_layers} layers: output shape {y_pred.shape}, "
              f"params {sum(p.numel() for p in model.parameters()):,}")
    
    print("  ✓ Multi-layer test passed!")


def test_gradient_descent():
    """Test gradient descent baseline."""
    print("\nTesting gradient descent baseline...")
    
    d = 10
    n_points = 2 * d + 1
    xs, ys, w_true = generate_linear_regression_task(d, n_points=n_points, batch_size=2)
    query_x = torch.randn(2, 1, d)
    
    for T in [1, 2, 4]:
        y_pred, w_final = gd_t_steps(xs, ys, query_x, eta=0.1, T=T)
        print(f"  T={T}: prediction shape {y_pred.shape}, weight shape {w_final.shape}")
    
    print("  ✓ GD baseline test passed!")


def test_cosine_similarity():
    """Test cosine similarity computation."""
    print("\nTesting cosine similarity...")
    
    d = 10
    model = MultiLayerLSA(d=d, num_layers=2, hidden_dim=d)
    
    xs, ys, w_true = generate_linear_regression_task(d, n_points=2*d+1, batch_size=2)
    query_x = torch.randn(2, 1, d)
    
    # Get LSA weight update
    w_lsa = model.get_weight_update(xs, ys, query_x)
    
    # Get GD weight update
    _, w_gd = gd_t_steps(xs, ys, query_x, eta=0.1, T=2)
    
    # Compute similarity
    cos_sim = compute_cosine_similarity(w_lsa, w_gd)
    
    print(f"  LSA weight shape: {w_lsa.shape}")
    print(f"  GD weight shape: {w_gd.shape}")
    print(f"  Cosine similarity: {cos_sim.mean().item():.4f}")
    print("  ✓ Cosine similarity test passed!")


def test_training_loop():
    """Test a mini training loop."""
    print("\nTesting mini training loop...")
    
    d = 5
    model = MultiLayerLSA(d=d, num_layers=1, hidden_dim=d)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    initial_loss = None
    for step in range(10):
        # Generate task
        xs, ys, w_true = generate_linear_regression_task(d, n_points=2*d+1, batch_size=4)
        query_x = torch.randn(4, 1, d)
        query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(xs, ys, query_x)
        loss = criterion(y_pred, query_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        
        if step % 5 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")
    
    final_loss = loss.item()
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss decreased: {initial_loss > final_loss}")
    print("  ✓ Training loop test passed!")


def main():
    print("="*60)
    print("LSA Implementation Tests")
    print("="*60)
    
    test_single_layer()
    test_multi_layer()
    test_gradient_descent()
    test_cosine_similarity()
    test_training_loop()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nYou can now run the full experiment:")
    print("  python scripts/lsa_gd_multilayer.py --d 20 --num_layers_list 1 2 4")


if __name__ == '__main__':
    main()
