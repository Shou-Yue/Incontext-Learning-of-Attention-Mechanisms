import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')

from config import ExperimentConfig
from models.transformer import Transformer
from data.generate_prompts import generate_batch, get_curriculum_dim
import matplotlib.pyplot as plt


def run_smoke_test(attn_type, rank=8):
    """
    Run a quick smoke test to verify mechanism trains without errors.
    
    Args:
        attn_type: 'baseline' | 'linear' | 'lowrank'
        rank: Rank parameter (for lowrank only)
    
    Returns:
        Dict with losses and cosine similarities
    """
    print(f"\n{'='*60}")
    print(f"Running smoke test: {attn_type}")
    print(f"{'='*60}")
    
    # Small config for speed
    config = ExperimentConfig(
        attn_type=attn_type,
        attn_rank=rank,
        task_dim=10,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_mlp=512,
        batch_size=32,
        learning_rate=1e-3,
        total_steps=2000,  # Quick test
        log_interval=200,
        curriculum_start_dim=5,
        curriculum_end_step=1000
    )
    
    # Build model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    losses = []
    cosine_sims = []
    steps = []
    
    for step in range(config.total_steps):
        current_dim = get_curriculum_dim(
            step, 
            config.curriculum_start_dim, 
            config.task_dim, 
            config.curriculum_end_step
        )
        
        # Generate batch
        batch = generate_batch(
            batch_size=config.batch_size,
            dim=current_dim,
            n_examples=10,
            noise_std=0.0
        )
        
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward
        predictions = model(inputs)
        
        # Loss on query token only (last token)
        loss = F.mse_loss(predictions[:, -1, :], targets[:, -1, :])
        
        # Check for NaNs
        if torch.isnan(loss):
            print(f"❌ FAILED: NaN loss at step {step}")
            return None
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        steps.append(step)
        
        # Compute cosine similarity every 200 steps
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                # Simple cosine sim computation
                eval_batch = generate_batch(8, current_dim, 10, 0.0)
                X = eval_batch['inputs'][:, ::2, :][:, :-1, :].to(device)  # Extract x values
                y = eval_batch['inputs'][:, 1::2, 0].to(device)  # Extract y values
                n_examples = X.shape[1]
                
                # GD-1 baseline
                w_GD = torch.einsum('bnd,bn->bd', X, y) / n_examples
                
                # Model prediction on canonical basis
                x_test = torch.eye(current_dim).unsqueeze(0).repeat(8, 1, 1).to(device)
                test_inputs = torch.zeros(8, 2*n_examples + 1, current_dim).to(device)
                for i in range(n_examples):
                    test_inputs[:, 2*i, :] = X[:, i, :]
                    test_inputs[:, 2*i + 1, 0] = y[:, i]
                
                preds = []
                for i in range(current_dim):
                    test_inputs[:, -1, :] = x_test[:, i, :]
                    pred = model(test_inputs)[:, -1, 0]
                    preds.append(pred)
                w_model = torch.stack(preds, dim=1)
                
                # Cosine similarity
                cos_sim = F.cosine_similarity(w_model, w_GD, dim=-1).mean().item()
                cosine_sims.append(cos_sim)
            
            model.train()
            print(f"  Step {step}, Loss: {loss.item():.4f}, Dim: {current_dim}, Cos Sim: {cos_sim:.3f}")
        elif step % config.log_interval == 0:
            print(f"  Step {step}, Loss: {loss.item():.4f}, Dim: {current_dim}")
    
    # Check convergence
    initial_loss = sum(losses[:100]) / 100
    final_loss = sum(losses[-100:]) / 100
    
    if final_loss < initial_loss:
        print(f"✓ PASSED: Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
        return {'losses': losses, 'cosine_sims': cosine_sims, 'steps': steps, 'success': True}
    else:
        print(f"❌ FAILED: Loss did not decrease ({initial_loss:.4f} → {final_loss:.4f})")
        return {'losses': losses, 'cosine_sims': cosine_sims, 'steps': steps, 'success': False}


def plot_results(results):
    """Generate plots from smoke test results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Learning curves
    for name, result in results.items():
        if result is not None:
            ax1.plot(result['steps'], result['losses'], label=name, alpha=0.7)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('MSE Loss (Query Token)')
    ax1.set_title('Learning Curves: Smoke Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Cosine similarity
    for name, result in results.items():
        if result is not None and len(result['cosine_sims']) > 0:
            cos_steps = [i * 200 for i in range(len(result['cosine_sims']))]
            ax2.plot(cos_steps, result['cosine_sims'], label=name, marker='o', alpha=0.7)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Cosine Similarity to GD-1')
    ax2.set_title('Alignment with Gradient Descent')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('../results/figures/smoke_test_results.png', dpi=150)
    print(f"\n✓ Plots saved to results/figures/smoke_test_results.png")
    plt.close()


def main():
    """Run smoke tests for all mechanisms."""
    print("SMOKE TEST: In-Context Learning Attention Mechanisms")
    print("=" * 60)
    
    results = {}
    
    # Test all three mechanisms
    for attn_type in ['baseline', 'linear', 'lowrank']:
        result = run_smoke_test(attn_type, rank=8)
        results[attn_type] = result
    
    # Print summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if result is not None:
            status = "✓ PASS" if result['success'] else "❌ FAIL"
            final_loss = result['losses'][-1]
            final_cos = result['cosine_sims'][-1] if len(result['cosine_sims']) > 0 else 0
            print(f"{name:15s} {status:10s} | Final Loss: {final_loss:.4f} | Cos Sim: {final_cos:.3f}")
        else:
            print(f"{name:15s} {'❌ FAIL':10s} | NaN encountered")
    
    all_passed = all(r['success'] for r in results.values() if r is not None)
    
    # Generate plots
    plot_results(results)
    
    if all_passed:
        print("\n✓ All smoke tests passed! Ready for full experiments.")
    else:
        print("\n⚠ Some smoke tests had issues. Review results above.")
    
    return all_passed


if __name__ == '__main__':
    import time
    start_time = time.time()
    success = main()
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    exit(0 if success else 1)
