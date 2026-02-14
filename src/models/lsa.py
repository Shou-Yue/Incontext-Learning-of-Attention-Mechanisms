"""
Linear Self-Attention (LSA) models for in-context learning.

Implements single and multi-layer LSA to approximate gradient descent.
Based on Oswald et al. (2023): "Transformers Learn In-Context by Gradient Descent"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSelfAttentionBlock(nn.Module):
    """
    Single Linear Self-Attention layer.
    
    Uses linear projections (no softmax) to implement one step of gradient descent
    on linear regression tasks.
    """
    
    def __init__(self, d, hidden_dim=None):
        """
        Args:
            d: Input dimension (feature dimension)
            hidden_dim: Hidden dimension for projections (default: d)
        """
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        
        # Token dimension is d+1 (d features + 1 label)
        self.token_dim = d + 1
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_k = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_v = nn.Linear(self.token_dim, self.token_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(self.token_dim, self.token_dim, bias=False)
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)
        
        # Initialize weights carefully
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.01)
    
    def forward(self, tokens):
        """
        Forward pass through LSA layer.
        
        Args:
            tokens: (batch_size, n_points, token_dim) where token_dim = d + 1
                    Each token is [x_i; y_i] where x_i ∈ R^d, y_i ∈ R
        
        Returns:
            output: (batch_size, n_points, token_dim) updated token representations
        """
        batch_size, n_points, _ = tokens.shape
        
        # Layer norm before attention
        tokens_norm = self.norm1(tokens)
        
        # Compute Q, K, V
        Q = self.W_q(tokens_norm)  # (batch_size, n_points, hidden_dim)
        K = self.W_k(tokens_norm)  # (batch_size, n_points, hidden_dim)
        V = self.W_v(tokens_norm)  # (batch_size, n_points, token_dim)
        
        # Linear attention: A = Q K^T (no softmax!)
        attn_weights = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, n_points, n_points)
        attn_weights = attn_weights / (self.hidden_dim ** 0.5)  # scaling
        
        # Causal mask: each token can only attend to previous tokens and itself
        mask = torch.tril(torch.ones(n_points, n_points, device=tokens.device))
        attn_weights = attn_weights.masked_fill(mask == 0, 0.0)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, V)  # (batch_size, n_points, token_dim)
        
        # Output projection with small scale
        output = self.W_o(attn_output) * 0.1  # (batch_size, n_points, token_dim)
        
        # Residual connection
        output = tokens + output
        
        # Layer norm after residual
        output = self.norm2(output)
        
        return output


class MultiLayerLSA(nn.Module):
    """
    Multi-layer Linear Self-Attention model.
    
    Stacks multiple LSA blocks to approximate multi-step gradient descent.
    When num_layers = T, this should approximate T steps of GD.
    """
    
    def __init__(self, d, num_layers=1, hidden_dim=None):
        """
        Args:
            d: Input dimension (feature dimension)
            num_layers: Number of LSA layers to stack
            hidden_dim: Hidden dimension for projections (default: d)
        """
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        
        # Stack of LSA blocks
        self.layers = nn.ModuleList([
            LinearSelfAttentionBlock(d, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Final prediction head (extracts y from last query token)
        self.pred_head = nn.Linear(d + 1, 1, bias=False)
        nn.init.xavier_uniform_(self.pred_head.weight, gain=0.1)
        
    def forward(self, xs, ys, query_x):
        """
        Forward pass through multi-layer LSA.
        
        Args:
            xs: (batch_size, n_points, d) training inputs
            ys: (batch_size, n_points) training outputs
            query_x: (batch_size, 1, d) query input to predict on
        
        Returns:
            y_pred: (batch_size,) predicted output for query
        """
        batch_size, n_points, d = xs.shape
        
        # Create tokens: [x_i; y_i] for training points
        # For training points: use actual y values
        train_tokens = torch.cat([xs, ys.unsqueeze(-1)], dim=-1)  # (batch_size, n_points, d+1)
        
        # For query point: use y=0 as placeholder
        query_y = torch.zeros(batch_size, 1, device=xs.device)
        query_token = torch.cat([query_x, query_y.unsqueeze(-1)], dim=-1)  # (batch_size, 1, d+1)
        
        # Concatenate training and query tokens
        tokens = torch.cat([train_tokens, query_token], dim=1)  # (batch_size, n_points+1, d+1)
        
        # Pass through all LSA layers
        for layer in self.layers:
            tokens = layer(tokens)
        
        # Extract last token (query) and predict
        query_token_out = tokens[:, -1, :]  # (batch_size, d+1)
        y_pred = self.pred_head(query_token_out)  # (batch_size, 1)
        
        return y_pred.squeeze(-1)  # (batch_size,)
    
    def get_weight_update(self, xs, ys, query_x):
        """
        Extract the implied weight update from LSA by evaluating on multiple test points.
        
        This uses least squares to recover the implicit weight vector from predictions.
        
        Args:
            xs: (batch_size, n_points, d) training inputs
            ys: (batch_size, n_points) training outputs  
            query_x: (batch_size, 1, d) query input (not used, for API compatibility)
        
        Returns:
            delta_w: (batch_size, d) implied weight update vector
        """
        batch_size, n_points, d = xs.shape
        
        with torch.no_grad():
            # Generate multiple test points
            n_test = min(d * 2, 100)  # Sample enough points
            test_xs = torch.randn(batch_size, n_test, d, device=xs.device)
            
            # Get predictions for all test points
            test_preds = []
            for i in range(n_test):
                x_i = test_xs[:, i:i+1, :]  # (batch_size, 1, d)
                y_pred = self.forward(xs, ys, x_i)
                test_preds.append(y_pred)
            
            test_preds = torch.stack(test_preds, dim=1)  # (batch_size, n_test)
            
            # Solve for weight: w = (X^T X)^{-1} X^T y
            # Using least squares to fit w such that test_xs @ w ≈ test_preds
            delta_w = torch.zeros(batch_size, d, device=xs.device)
            for b in range(batch_size):
                X = test_xs[b]  # (n_test, d)
                y = test_preds[b]  # (n_test,)
                try:
                    w = torch.linalg.lstsq(X, y).solution
                    delta_w[b] = w
                except:
                    # Fallback to pseudoinverse
                    w = torch.linalg.pinv(X) @ y
                    delta_w[b] = w
        
        return delta_w
