import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class BaselineAttention(nn.Module):
    """
    Standard multi-head softmax attention (Vaswani et al., 2017).
    
    Complexity:
    - Time: O(n² · d_k) where n = sequence length, d_k = head dimension
    - Memory: O(n²) for attention matrix
    
    This serves as the baseline/control condition for our experiments.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n, d_model)
            mask: Optional (batch, n, n) boolean mask
        
        Returns:
            output: (batch, n, d_model)
        """
        batch, n, d = x.shape
        
        # Project to Q, K, V and reshape for multi-head
        Q = self.W_Q(x).view(batch, n, self.n_heads, self.d_k)  # (batch, n, n_heads, d_k)
        K = self.W_K(x).view(batch, n, self.n_heads, self.d_k)
        V = self.W_V(x).view(batch, n, self.n_heads, self.d_k)
        
        # Compute attention scores
        # scores: (batch, n, n_heads, n)
        scores = torch.einsum('bnhd,bmhd->bnhm', Q, K) / math.sqrt(self.d_k)
        
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
