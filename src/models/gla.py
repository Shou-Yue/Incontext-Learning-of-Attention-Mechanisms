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

    def __init__(self, d, hidden_dim=None, eps=1e-6, disable_gate: bool = False, gate_bias: float = 0.0):
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        self.token_dim = d + 1
        self.eps = eps
        self.disable_gate = disable_gate
        self.gate_bias = gate_bias

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
        nn.init.xavier_uniform_(self.W_g.weight, gain=0.02)
        nn.init.constant_(self.W_g.bias, self.gate_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm1(tokens)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Gate values
        if self.disable_gate:
            Vg = V
        else:
            G = torch.sigmoid(self.W_g(x))
            Vg = V * G

        # Kernel feature map
        Qp = phi(Q)
        Kp = phi(K)

        kv = torch.einsum("bth,btm->bthm", Kp, Vg)
        S_kv = kv.cumsum(dim=1)
        S_k = Kp.cumsum(dim=1)

        numerator = torch.einsum("bth,bthm->btm", Qp, S_kv)
        denom = torch.einsum("bth,bth->bt", Qp, S_k).clamp_min(self.eps)
        attn_out = numerator / denom.unsqueeze(-1)

        out = self.W_o(attn_out) * 0.1
        out = tokens + out
        out = self.norm2(out)
        return out


class MultiLayerGLA(nn.Module):
    """
    Multi-layer Gated Linear Attention model.
    """

    def __init__(self, d, num_layers=1, hidden_dim=None, disable_gate: bool = False, gate_bias: float = 0.0):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else d

        self.layers = nn.ModuleList([
            GatedLinearAttentionBlock(
                d,
                hidden_dim=self.hidden_dim,
                disable_gate=disable_gate,
                gate_bias=gate_bias,
            )
            for _ in range(num_layers)
        ])

        self.pred_head = nn.Linear(d + 1, 1, bias=False)
        nn.init.xavier_uniform_(self.pred_head.weight, gain=0.1)

    def forward(self, xs, ys, query_x):
        batch_size, n_points, d = xs.shape

        train_tokens = torch.cat([xs, ys.unsqueeze(-1)], dim=-1)
        query_y = torch.zeros(batch_size, 1, device=xs.device, dtype=xs.dtype)
        query_token = torch.cat([query_x, query_y.unsqueeze(-1)], dim=-1)
        tokens = torch.cat([train_tokens, query_token], dim=1)

        for layer in self.layers:
            tokens = layer(tokens)

        query_token_out = tokens[:, -1, :]
        y_pred = self.pred_head(query_token_out)
        return y_pred.squeeze(-1)

    def get_weight_update(self, xs, ys, query_x):
        batch_size, n_points, d = xs.shape

        with torch.no_grad():
            n_test = min(d * 2, 100)
            test_xs = torch.randn(batch_size, n_test, d, device=xs.device, dtype=xs.dtype)

            test_preds = []
            for i in range(n_test):
                x_i = test_xs[:, i:i+1, :]
                y_pred = self.forward(xs, ys, x_i)
                test_preds.append(y_pred)

            test_preds = torch.stack(test_preds, dim=1)

            delta_w = torch.zeros(batch_size, d, device=xs.device, dtype=xs.dtype)
            for b in range(batch_size):
                X = test_xs[b]
                y = test_preds[b]
                try:
                    w = torch.linalg.lstsq(X, y).solution
                except Exception:
                    w = torch.linalg.pinv(X) @ y
                delta_w[b] = w

        return delta_w
