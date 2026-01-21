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
