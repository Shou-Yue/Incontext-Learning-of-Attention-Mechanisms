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
