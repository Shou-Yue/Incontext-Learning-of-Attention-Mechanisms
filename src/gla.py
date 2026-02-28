"""
Gated Linear Attention (GLA) models for in-context learning.

Causal linear attention:
  S_k(t)  = sum_{i<=t}  φ(k_i)
  S_kv(t) = sum_{i<=t}  φ(k_i) ⊗ (v_i ⊙ g_i)
  out_t   = ( φ(q_t) · S_kv(t) ) / ( φ(q_t) · S_k(t) + eps )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def phi(x: torch.Tensor) -> torch.Tensor:
    # Common positive feature map for linear attention
    return F.elu(x) + 1.0


class GatedLinearAttentionBlock(nn.Module):
    """
    Single Gated Linear Attention layer (causal).

    Uses kernelized linear attention with a value gate:
        v_i <- v_i * sigmoid(W_g x_i)
    """

    def __init__(self, d, hidden_dim=None, eps=1e-6):
        """
        Args:
            d: Input dimension (feature dimension)
            hidden_dim: Hidden dimension for projections (default: d)
            eps: Numerical stability term for normalization
        """
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        self.token_dim = d + 1
        self.eps = eps

        # Projections
        self.W_q = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_k = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_v = nn.Linear(self.token_dim, self.token_dim, bias=False)

        # Value gate (same dim as V)
        self.W_g = nn.Linear(self.token_dim, self.token_dim, bias=True)

        # Output projection
        self.W_o = nn.Linear(self.token_dim, self.token_dim, bias=False)

        # Norms (pre-norm + post-residual norm)
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.02)
        # Gate bias can help start near "open" or "closed"; keep near neutral.
        nn.init.xavier_uniform_(self.W_g.weight, gain=0.02)
        nn.init.zeros_(self.W_g.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, n_tokens, token_dim)

        Returns:
            (batch_size, n_tokens, token_dim)
        """
        x = self.norm1(tokens)  # pre-norm

        Q = self.W_q(x)  # (b, t, h)
        K = self.W_k(x)  # (b, t, h)
        V = self.W_v(x)  # (b, t, m) where m=token_dim

        # Gate values
        G = torch.sigmoid(self.W_g(x))          # (b, t, m)
        Vg = V * G                              # (b, t, m)

        # Kernel feature map (positive)
        Qp = phi(Q)                             # (b, t, h)
        Kp = phi(K)                             # (b, t, h)

        # Causal linear attention via prefix sums:
        # S_k(t)  = Σ_{i<=t} Kp_i
        # S_kv(t) = Σ_{i<=t} Kp_i ⊗ Vg_i
        # out_t   = (Qp_t · S_kv(t)) / (Qp_t · S_k(t) + eps)
        #
        # kv: (b, t, h, m)
        kv = torch.einsum("bth,btm->bthm", Kp, Vg)
        S_kv = kv.cumsum(dim=1)                 # (b, t, h, m)
        S_k = Kp.cumsum(dim=1)                  # (b, t, h)

        numerator = torch.einsum("bth,bthm->btm", Qp, S_kv)     # (b, t, m)
        denom = torch.einsum("bth,bth->bt", Qp, S_k).clamp_min(self.eps)  # (b, t)
        attn_out = numerator / denom.unsqueeze(-1)              # (b, t, m)

        out = self.W_o(attn_out) * 0.1
        out = tokens + out
        out = self.norm2(out)
        return out


class MultiLayerGLA(nn.Module):
    """
    Multi-layer Gated Linear Attention model.

    Stacks multiple GLA blocks (each is a causal linear-attention layer with gating).
    """

    def __init__(self, d, num_layers=1, hidden_dim=None):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else d

        self.layers = nn.ModuleList([
            GatedLinearAttentionBlock(d, hidden_dim=self.hidden_dim)
            for _ in range(num_layers)
        ])

        self.pred_head = nn.Linear(d + 1, 1, bias=False)
        nn.init.xavier_uniform_(self.pred_head.weight, gain=0.1)

    def forward(self, xs, ys, query_x):
        """
        Args:
            xs: (batch_size, n_points, d)
            ys: (batch_size, n_points)
            query_x: (batch_size, 1, d)

        Returns:
            (batch_size,)
        """
        batch_size, n_points, d = xs.shape

        train_tokens = torch.cat([xs, ys.unsqueeze(-1)], dim=-1)  # (b, n, d+1)

        query_y = torch.zeros(batch_size, 1, device=xs.device, dtype=xs.dtype)
        query_token = torch.cat([query_x, query_y.unsqueeze(-1)], dim=-1)  # (b, 1, d+1)

        tokens = torch.cat([train_tokens, query_token], dim=1)  # (b, n+1, d+1)

        for layer in self.layers:
            tokens = layer(tokens)

        query_token_out = tokens[:, -1, :]          # (b, d+1)
        y_pred = self.pred_head(query_token_out)    # (b, 1)
        return y_pred.squeeze(-1)

    def get_weight_update(self, xs, ys, query_x):
        """
        Same helper as before: recover implied linear weights by probing predictions.
        """
        batch_size, n_points, d = xs.shape

        with torch.no_grad():
            n_test = min(d * 2, 100)
            test_xs = torch.randn(batch_size, n_test, d, device=xs.device, dtype=xs.dtype)

            test_preds = []
            for i in range(n_test):
                x_i = test_xs[:, i:i+1, :]
                y_pred = self.forward(xs, ys, x_i)
                test_preds.append(y_pred)

            test_preds = torch.stack(test_preds, dim=1)  # (b, n_test)

            delta_w = torch.zeros(batch_size, d, device=xs.device, dtype=xs.dtype)
            for b in range(batch_size):
                X = test_xs[b]     # (n_test, d)
                y = test_preds[b]  # (n_test,)
                try:
                    w = torch.linalg.lstsq(X, y).solution
                except Exception:
                    w = torch.linalg.pinv(X) @ y
                delta_w[b] = w

        return delta_w