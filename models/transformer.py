import torch
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
    """Full transformer model for in-context learning."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Input embedding (use max dimension to handle curriculum)
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
            x: (batch, seq_len, dim) where dim can vary during curriculum
        
        Returns:
            predictions: (batch, seq_len, 1)
        """
        batch, seq_len, current_dim = x.shape
        
        # Pad to task_dim if current_dim is smaller (curriculum learning)
        if current_dim < self.config.task_dim:
            padding = torch.zeros(batch, seq_len, self.config.task_dim - current_dim).to(x.device)
            x = torch.cat([x, padding], dim=-1)
        
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_proj(x)
