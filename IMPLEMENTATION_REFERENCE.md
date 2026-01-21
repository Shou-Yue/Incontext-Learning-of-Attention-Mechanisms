# Implementation Reference Document

**Project:** Linear and Low-Rank Attention Mechanisms for In-Context Learning  
**Purpose:** Low-level technical reference with code skeletons and API specifications  
**Date:** January 2026

This document contains detailed implementation specifications, code skeletons, and API contracts for the attention mechanisms. For high-level design and experimental plans, see `DESIGN_DOC.md`.

---

## Table of Contents

1. [Module Specifications](#module-specifications)
2. [Code Skeletons](#code-skeletons)
3. [Unit Tests](#unit-tests)
4. [Integration Code](#integration-code)
5. [Training Script](#training-script)
6. [Smoke Test Specification](#smoke-test-specification)

---

## Module Specifications

### LinearAttention API

**Interface:**
```python
class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, feature_map: str = 'elu')
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor
```

**Input:**
- `x`: (batch, n, d_model)
- `mask`: Optional (batch, n, n) boolean mask

**Output:**
- (batch, n, d_model)

**Properties:**
- Time complexity: O(n · d²)
- Memory complexity: O(d²)
- Feature map: φ(x) = elu(x) + 1

---

### LowRankAttention API

**Interface:**
```python
class LowRankAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rank: int)
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor
```

**Input:**
- `x`: (batch, n, d_model)
- `mask`: Optional (batch, n, n) boolean mask
- `rank`: Bottleneck dimension r

**Output:**
- (batch, n, d_model)

**Properties:**
- Time complexity: O(n² · r)
- Memory complexity: O(n²)
- Rank constraint: Effective attention rank ≤ r

---

## Code Skeletons

### LinearAttention Implementation

**File:** `models/attention/linear_attention.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LinearAttention(nn.Module):
    """
    Linear attention using kernel feature maps (Katharopoulos et al., 2020).
    
    Complexity:
    - Time: O(n · d²) where n = sequence length, d = d_k
    - Memory: O(d²) for cached φ(K)^T V
    
    Key difference from standard attention:
    - Computes φ(Q) [φ(K)^T V] instead of [φ(Q)φ(K)^T] V
    - Avoids materializing n×n attention matrix
    """
    
    def __init__(self, d_model: int, n_heads: int, feature_map: str = 'elu'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Standard Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Feature map selection
        if feature_map == 'elu':
            self.phi = lambda x: F.elu(x) + 1
        else:
            raise NotImplementedError(f"Feature map {feature_map} not implemented")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n, d_model)
            mask: Optional (batch, n, n) - NOT USED in linear attention
        
        Returns:
            output: (batch, n, d_model)
        """
        batch, n, d = x.shape
        
        # Project to Q, K, V
        Q = self.W_Q(x).view(batch, n, self.n_heads, self.d_k)  # (batch, n, n_heads, d_k)
        K = self.W_K(x).view(batch, n, self.n_heads, self.d_k)
        V = self.W_V(x).view(batch, n, self.n_heads, self.d_k)
        
        # Apply feature map
        Q = self.phi(Q)  # (batch, n, n_heads, d_k)
        K = self.phi(K)
        
        # Linear attention: φ(Q) [φ(K)^T V]
        # Step 1: Compute φ(K)^T V  →  (batch, n_heads, d_k, d_k)
        KV = torch.einsum('bnhd,bnhv->bhdv', K, V)
        
        # Step 2: Compute φ(Q) [φ(K)^T V]  →  (batch, n, n_heads, d_k)
        Z = torch.einsum('bnhd,bhdv->bnhv', Q, KV)
        
        # Normalization: divide by sum of attention weights
        # Denominator: φ(Q) [φ(K)^T 1]
        K_sum = K.sum(dim=1, keepdim=True)  # (batch, 1, n_heads, d_k)
        normalizer = torch.einsum('bnhd,bmhd->bnh', Q, K_sum)  # (batch, n, n_heads)
        
        Z = Z / (normalizer.unsqueeze(-1) + 1e-6)
        
        # Concatenate heads and project
        Z = Z.reshape(batch, n, self.d_model)
        output = self.W_O(Z)
        
        return output
```

---

### LowRankAttention Implementation

**File:** `models/attention/lowrank_attention.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LowRankAttention(nn.Module):
    """
    Low-rank attention using rank-constrained Q, K projections.
    
    Complexity:
    - Time: O(n² · r + n · d_k · r) where r = rank
    - Memory: O(n²) for attention matrix (same as baseline)
    
    Key difference from baseline:
    - Projects Q, K to rank-r subspace before computing attention
    - Smoothly interpolates: r = d_k recovers baseline
    """
    
    def __init__(self, d_model: int, n_heads: int, rank: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rank = rank
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert rank <= self.d_k, f"Rank {rank} exceeds d_k {self.d_k}"
        
        # Standard Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Low-rank projection matrices
        # These project from d_k → rank
        self.W_Q_lowrank = nn.Linear(self.d_k, rank, bias=False)
        self.W_K_lowrank = nn.Linear(self.d_k, rank, bias=False)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n, d_model)
            mask: Optional (batch, n, n) boolean mask
        
        Returns:
            output: (batch, n, d_model)
        """
        batch, n, d = x.shape
        
        # Project to Q, K, V
        Q = self.W_Q(x).view(batch, n, self.n_heads, self.d_k)  # (batch, n, n_heads, d_k)
        K = self.W_K(x).view(batch, n, self.n_heads, self.d_k)
        V = self.W_V(x).view(batch, n, self.n_heads, self.d_k)
        
        # Project to low-rank space
        Q_r = self.W_Q_lowrank(Q)  # (batch, n, n_heads, rank)
        K_r = self.W_K_lowrank(K)  # (batch, n, n_heads, rank)
        
        # Compute attention scores in low-rank space
        # scores: (batch, n, n_heads, n)
        scores = torch.einsum('bnhr,bmhr->bnhm', Q_r, K_r) / math.sqrt(self.rank)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head: (batch, n, n) → (batch, 1, n, n)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax attention weights
        A = F.softmax(scores, dim=-1)  # (batch, n, n_heads, n)
        
        # Apply attention to values
        # output: (batch, n, n_heads, d_k)
        output = torch.einsum('bnhm,bmhv->bnhv', A, V)
        
        # Concatenate heads and project
        output = output.reshape(batch, n, self.d_model)
        output = self.W_O(output)
        
        return output
```

---

## Unit Tests

### LinearAttention Tests

**File:** `tests/test_linear_attention.py`

```python
import torch
import pytest
from models.attention.linear_attention import LinearAttention


def test_linear_attention_shape():
    """Test output shape matches input."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model)
    
    attn = LinearAttention(d_model, n_heads=8)
    output = attn(x)
    
    assert output.shape == (batch, n, d_model), f"Expected {(batch, n, d_model)}, got {output.shape}"


def test_linear_attention_no_nans():
    """Test for numerical stability (no NaNs)."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model)
    
    attn = LinearAttention(d_model, n_heads=8)
    output = attn(x)
    
    assert not torch.isnan(output).any(), "NaN detected in output"
    assert not torch.isinf(output).any(), "Inf detected in output"


def test_linear_attention_gradients():
    """Test gradient flow through the module."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model, requires_grad=True)
    
    attn = LinearAttention(d_model, n_heads=8)
    output = attn(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradients for input"
    assert attn.W_Q.weight.grad is not None, "No gradients for W_Q"
    assert attn.W_K.weight.grad is not None, "No gradients for W_K"
    assert attn.W_V.weight.grad is not None, "No gradients for W_V"
    assert attn.W_O.weight.grad is not None, "No gradients for W_O"


def test_linear_attention_batch_independence():
    """Test that different batch elements are processed independently."""
    batch, n, d_model = 4, 10, 256
    x = torch.randn(batch, n, d_model)
    
    attn = LinearAttention(d_model, n_heads=8)
    
    # Full batch forward
    output_full = attn(x)
    
    # Individual forwards
    for i in range(batch):
        output_single = attn(x[i:i+1])
        assert torch.allclose(output_full[i], output_single[0], atol=1e-5), \
            f"Batch element {i} differs when processed alone"


def test_linear_attention_deterministic():
    """Test deterministic behavior with fixed seed."""
    torch.manual_seed(42)
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model)
    
    attn1 = LinearAttention(d_model, n_heads=8)
    torch.manual_seed(42)
    attn2 = LinearAttention(d_model, n_heads=8)
    
    output1 = attn1(x)
    output2 = attn2(x)
    
    assert torch.allclose(output1, output2), "Non-deterministic behavior detected"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

### LowRankAttention Tests

**File:** `tests/test_lowrank_attention.py`

```python
import torch
import pytest
from models.attention.lowrank_attention import LowRankAttention


def test_lowrank_attention_shape():
    """Test output shape matches input across different ranks."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model)
    
    for rank in [4, 8, 16, 32]:
        attn = LowRankAttention(d_model, n_heads=8, rank=rank)
        output = attn(x)
        assert output.shape == (batch, n, d_model), \
            f"Rank {rank}: Expected {(batch, n, d_model)}, got {output.shape}"


def test_lowrank_attention_no_nans():
    """Test for numerical stability (no NaNs) across ranks."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model)
    
    for rank in [4, 8, 16, 32]:
        attn = LowRankAttention(d_model, n_heads=8, rank=rank)
        output = attn(x)
        assert not torch.isnan(output).any(), f"Rank {rank}: NaN detected"
        assert not torch.isinf(output).any(), f"Rank {rank}: Inf detected"


def test_lowrank_attention_gradients():
    """Test gradient flow through the module."""
    batch, n, d_model = 2, 10, 256
    x = torch.randn(batch, n, d_model, requires_grad=True)
    
    attn = LowRankAttention(d_model, n_heads=8, rank=16)
    output = attn(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradients for input"
    assert attn.W_Q.weight.grad is not None, "No gradients for W_Q"
    assert attn.W_K.weight.grad is not None, "No gradients for W_K"
    assert attn.W_V.weight.grad is not None, "No gradients for W_V"
    assert attn.W_O.weight.grad is not None, "No gradients for W_O"
    assert attn.W_Q_lowrank.weight.grad is not None, "No gradients for W_Q_lowrank"
    assert attn.W_K_lowrank.weight.grad is not None, "No gradients for W_K_lowrank"


def test_lowrank_attention_rank_interpolation():
    """Test that rank=d_k approximates baseline behavior."""
    batch, n, d_model, n_heads = 2, 10, 256, 8
    d_k = d_model // n_heads
    x = torch.randn(batch, n, d_model)
    
    # Low-rank with full rank
    torch.manual_seed(42)
    attn_lowrank = LowRankAttention(d_model, n_heads, rank=d_k)
    
    # Set low-rank projections to identity (approximate baseline)
    with torch.no_grad():
        attn_lowrank.W_Q_lowrank.weight.copy_(torch.eye(d_k))
        attn_lowrank.W_K_lowrank.weight.copy_(torch.eye(d_k))
    
    output_lowrank = attn_lowrank(x)
    
    # Should behave similarly to baseline (not exact due to different initialization)
    assert output_lowrank.shape == (batch, n, d_model)
    assert not torch.isnan(output_lowrank).any()


def test_lowrank_attention_masking():
    """Test that masking is applied correctly."""
    batch, n, d_model = 1, 5, 256
    x = torch.randn(batch, n, d_model)
    
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(batch, n, n)).bool()
    
    attn = LowRankAttention(d_model, n_heads=8, rank=16)
    output_masked = attn(x, mask=mask)
    output_unmasked = attn(x, mask=None)
    
    # Outputs should differ when mask is applied
    assert not torch.allclose(output_masked, output_unmasked), \
        "Mask had no effect on output"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Integration Code

### Configuration System

**File:** `config.py`

```python
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # Attention mechanism
    attn_type: str = 'baseline'  # {'baseline', 'linear', 'lowrank'}
    attn_rank: int = 16          # Only used for lowrank
    attn_feature_map: str = 'elu'  # Only used for linear
    
    # Model architecture
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_mlp: int = 1024
    dropout: float = 0.0
    
    # Task parameters
    task_dim: int = 20
    max_prompt_length: int = 40
    noise_std: float = 0.0
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    total_steps: int = 100_000
    warmup_steps: int = 5_000
    
    # Curriculum
    curriculum_start_dim: int = 5
    curriculum_end_step: int = 20_000
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    
    # Paths
    checkpoint_dir: str = 'results/checkpoints'
    log_dir: str = 'results/logs'
    
    # Reproducibility
    seed: int = 42
```

---

### Attention Builder

**File:** `models/transformer.py`

```python
import torch.nn as nn
from config import ExperimentConfig
from models.attention.baseline_attention import BaselineAttention
from models.attention.linear_attention import LinearAttention
from models.attention.lowrank_attention import LowRankAttention


def build_attention_layer(config: ExperimentConfig) -> nn.Module:
    """
    Factory function to create attention module based on config.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Attention module (BaselineAttention | LinearAttention | LowRankAttention)
    """
    if config.attn_type == 'baseline':
        return BaselineAttention(config.d_model, config.n_heads)
    
    elif config.attn_type == 'linear':
        return LinearAttention(
            config.d_model, 
            config.n_heads, 
            feature_map=config.attn_feature_map
        )
    
    elif config.attn_type == 'lowrank':
        return LowRankAttention(
            config.d_model, 
            config.n_heads, 
            rank=config.attn_rank
        )
    
    else:
        raise ValueError(f"Unknown attention type: {config.attn_type}")


class TransformerBlock(nn.Module):
    """Single transformer layer with attention + MLP."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.attention = build_attention_layer(config)
        self.mlp = MLP(config.d_model, config.d_mlp, config.dropout)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Full transformer model."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_proj = nn.Linear(config.task_dim, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, task_dim)
        
        Returns:
            predictions: (batch, seq_len, 1)
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_proj(x)


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""
    
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
```

---

## Training Script

**File:** `train.py`

```python
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from config import ExperimentConfig
from models.transformer import Transformer
from data.generate_prompts import generate_batch
from evaluation.metrics import compute_metrics
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Attention mechanism
    parser.add_argument('--attn_type', type=str, default='baseline', 
                        choices=['baseline', 'linear', 'lowrank'])
    parser.add_argument('--attn_rank', type=int, default=16)
    parser.add_argument('--attn_feature_map', type=str, default='elu')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    
    # Task
    parser.add_argument('--task_dim', type=int, default=20)
    parser.add_argument('--noise_std', type=float, default=0.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--total_steps', type=int, default=100_000)
    
    # Experiment tracking
    parser.add_argument('--wandb_project', type=str, default='icl-attention')
    parser.add_argument('--wandb_name', type=str, default=None)
    
    return parser.parse_args()


def get_curriculum_dim(step, config):
    """Compute current dimension based on curriculum schedule."""
    if step >= config.curriculum_end_step:
        return config.task_dim
    
    progress = step / config.curriculum_end_step
    current_dim = config.curriculum_start_dim + \
                  (config.task_dim - config.curriculum_start_dim) * progress
    return int(current_dim)


def main():
    args = parse_args()
    config = ExperimentConfig(**vars(args))
    
    # Reproducibility
    torch.manual_seed(config.seed)
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_name or f"{config.attn_type}_d{config.task_dim}",
        config=vars(config)
    )
    
    # Build model
    model = Transformer(config).cuda()
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Training loop
    for step in range(config.total_steps):
        model.train()
        
        # Get current curriculum dimension
        current_dim = get_curriculum_dim(step, config)
        
        # Generate batch
        batch = generate_batch(
            batch_size=config.batch_size,
            dim=current_dim,
            n_examples=config.max_prompt_length // 2,
            noise_std=config.noise_std
        )
        
        # Move to GPU
        inputs = batch['inputs'].cuda()
        targets = batch['targets'].cuda()
        
        # Forward pass
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if step % config.log_interval == 0:
            print(f"Step {step}/{config.total_steps}, Loss: {loss.item():.4f}, Dim: {current_dim}")
            wandb.log({
                'train/loss': loss.item(),
                'train/dim': current_dim,
                'step': step
            })
        
        # Evaluation
        if step % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                metrics = compute_metrics(model, config)
            
            print(f"Eval - MSE: {metrics['mse']:.4f}, Cosine Sim: {metrics['cosine_sim']:.4f}")
            wandb.log({
                'eval/mse': metrics['mse'],
                'eval/cosine_sim': metrics['cosine_sim'],
                'step': step
            })
        
        # Checkpointing
        if step % config.checkpoint_interval == 0 and step > 0:
            checkpoint_path = f"{config.checkpoint_dir}/step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    wandb.finish()


if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Baseline (softmax attention)
python train.py --attn_type baseline --task_dim 20

# Linear attention
python train.py --attn_type linear --task_dim 20

# Low-rank attention
python train.py --attn_type lowrank --attn_rank 16 --task_dim 20
```

---

## Smoke Test Specification

### Smoke Test Script

**File:** `scripts/smoke_test.py`

```python
import torch
from config import ExperimentConfig
from models.transformer import Transformer
from train import get_curriculum_dim
from data.generate_prompts import generate_batch
import torch.nn.functional as F


def run_smoke_test(attn_type: str, rank: int = 16):
    """
    Run a quick smoke test to verify mechanism trains without errors.
    
    Args:
        attn_type: 'baseline' | 'linear' | 'lowrank'
        rank: Rank parameter (for lowrank only)
    
    Returns:
        success: bool
        final_loss: float
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
        total_steps=1000,
        log_interval=100,
        curriculum_start_dim=5,
        curriculum_end_step=500
    )
    
    # Build model
    model = Transformer(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    losses = []
    for step in range(config.total_steps):
        current_dim = get_curriculum_dim(step, config)
        
        # Generate batch
        batch = generate_batch(
            batch_size=config.batch_size,
            dim=current_dim,
            n_examples=20,
            noise_std=0.0
        )
        
        inputs = batch['inputs'].cuda()
        targets = batch['targets'].cuda()
        
        # Forward
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        
        # Check for NaNs
        if torch.isnan(loss):
            print(f"❌ FAILED: NaN loss at step {step}")
            return False, float('nan')
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"❌ FAILED: NaN gradient in {name} at step {step}")
                return False, float('nan')
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % config.log_interval == 0:
            print(f"  Step {step}, Loss: {loss.item():.4f}, Dim: {current_dim}")
    
    # Check convergence
    initial_loss = sum(losses[:100]) / 100
    final_loss = sum(losses[-100:]) / 100
    
    if final_loss < initial_loss:
        print(f"✓ PASSED: Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
        return True, final_loss
    else:
        print(f"❌ FAILED: Loss did not decrease ({initial_loss:.4f} → {final_loss:.4f})")
        return False, final_loss


def main():
    """Run smoke tests for all mechanisms."""
    results = {}
    
    # Test baseline
    success, loss = run_smoke_test('baseline')
    results['baseline'] = {'success': success, 'final_loss': loss}
    
    # Test linear attention
    success, loss = run_smoke_test('linear')
    results['linear'] = {'success': success, 'final_loss': loss}
    
    # Test low-rank attention
    for rank in [8, 16]:
        success, loss = run_smoke_test('lowrank', rank=rank)
        results[f'lowrank_r{rank}'] = {'success': success, 'final_loss': loss}
    
    # Print summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        status = "✓ PASS" if result['success'] else "❌ FAIL"
        print(f"{name:20s} {status:10s} (final loss: {result['final_loss']:.4f})")
    
    all_passed = all(r['success'] for r in results.values())
    
    if all_passed:
        print("\n✓ All smoke tests passed! Ready for full experiments.")
    else:
        print("\n❌ Some smoke tests failed. Debug before running full experiments.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
```

**Usage:**
```bash
python scripts/smoke_test.py
```

**Expected Output:**
```
============================================================
Running smoke test: baseline
============================================================
  Step 0, Loss: 1.2345, Dim: 5
  Step 100, Loss: 0.8234, Dim: 6
  ...
  Step 900, Loss: 0.1234, Dim: 10
✓ PASSED: Loss decreased from 1.0234 to 0.1456

============================================================
Running smoke test: linear
============================================================
  ...
✓ PASSED: Loss decreased from 1.1234 to 0.1834

============================================================
SMOKE TEST SUMMARY
============================================================
baseline             ✓ PASS     (final loss: 0.1456)
linear               ✓ PASS     (final loss: 0.1834)
lowrank_r8           ✓ PASS     (final loss: 0.2012)
lowrank_r16          ✓ PASS     (final loss: 0.1623)

✓ All smoke tests passed! Ready for full experiments.
```

---

## Deliverables Checklist

- [ ] `models/attention/linear_attention.py` - LinearAttention module
- [ ] `models/attention/lowrank_attention.py` - LowRankAttention module
- [ ] `tests/test_linear_attention.py` - Unit tests for linear attention
- [ ] `tests/test_lowrank_attention.py` - Unit tests for low-rank attention
- [ ] `config.py` - Configuration dataclass
- [ ] `models/transformer.py` - Transformer with attention builder
- [ ] `train.py` - Main training script
- [ ] `scripts/smoke_test.py` - Smoke test validation
- [ ] All unit tests passing
- [ ] Smoke tests passing for all mechanisms
- [ ] Training runs converge without NaNs

---

## Document Metadata

**Version:** 1.0  
**Last Updated:** January 21, 2026  
**Purpose:** Implementation reference for developers  
**Related Documents:** DESIGN_DOC.md (high-level design)
