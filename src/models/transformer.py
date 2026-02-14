"""
Transformer model for in-context learning.
Based on GPT-2 architecture.
"""
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class InContextTransformer(nn.Module):
    """
    Transformer model for in-context learning of function classes.
    Takes input-output pairs and predicts outputs for new inputs.
    """
    
    def __init__(self, n_dims, n_positions, n_embd=256, n_layer=12, n_head=8):
        """
        Args:
            n_dims: Input/output dimension
            n_positions: Maximum sequence length (number of in-context examples)
            n_embd: Embedding dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
        """
        super().__init__()
        
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd
        
        # GPT-2 configuration
        config = GPT2Config(
            n_positions=2 * n_positions,  # x and y for each point
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,  # No dropout
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        
        # Input projection: maps n_dims to n_embd
        self._read_in = nn.Linear(n_dims, n_embd)
        
        # Transformer backbone
        self._backbone = GPT2Model(config)
        
        # Output projection: maps n_embd to 1 (scalar output)
        self._read_out = nn.Linear(n_embd, 1)
    
    @staticmethod
    def _combine(xs, ys):
        """
        Interleave inputs (xs) and outputs (ys) into a single sequence.
        
        Args:
            xs: Input tensor of shape (batch_size, n_points, n_dims)
            ys: Output tensor of shape (batch_size, n_points)
        
        Returns:
            zs: Combined tensor of shape (batch_size, 2 * n_points, n_dims)
                Format: [x1, y1, x2, y2, ..., xk, yk]
                where yi is padded with zeros to match dimension of xi
        """
        batch_size, n_points, n_dims = xs.shape
        
        # Pad ys to match dimension of xs
        # ys has shape (batch_size, n_points)
        # We want (batch_size, n_points, n_dims) where last n_dims-1 entries are 0
        ys_wide = torch.cat([
            ys.unsqueeze(-1),  # (batch_size, n_points, 1)
            torch.zeros(batch_size, n_points, n_dims - 1, device=ys.device)
        ], dim=-1)
        
        # Stack xs and ys_wide and interleave
        # Stack gives (batch_size, n_points, 2, n_dims)
        # View as (batch_size, 2 * n_points, n_dims)
        zs = torch.stack([xs, ys_wide], dim=2)
        zs = zs.view(batch_size, 2 * n_points, n_dims)
        
        return zs
    
    def forward(self, xs, ys, inds=None):
        """
        Forward pass of the model.
        
        Args:
            xs: Input tensor of shape (batch_size, n_points, n_dims)
            ys: Output tensor of shape (batch_size, n_points)
            inds: Indices at which to make predictions (default: all positions)
        
        Returns:
            predictions: Predicted outputs of shape (batch_size, len(inds))
        """
        if inds is None:
            inds = list(range(ys.shape[1]))
        
        # Combine xs and ys into interleaved sequence
        zs = self._combine(xs, ys)  # (batch_size, 2 * n_points, n_dims)
        
        # Project to embedding space
        embeds = self._read_in(zs)  # (batch_size, 2 * n_points, n_embd)
        
        # Pass through transformer
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        # output shape: (batch_size, 2 * n_points, n_embd)
        
        # Project to scalar output
        predictions = self._read_out(output)  # (batch_size, 2 * n_points, 1)
        predictions = predictions.squeeze(-1)  # (batch_size, 2 * n_points)
        
        # Extract predictions at y positions (every other position starting from 1)
        # Positions 1, 3, 5, ... correspond to y1, y2, y3, ...
        # At these positions, the model has seen the corresponding x due to causal masking
        predictions = predictions[:, 1::2]  # (batch_size, n_points)
        
        # Return predictions at specified indices
        predictions = predictions[:, inds]
        
        return predictions
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
