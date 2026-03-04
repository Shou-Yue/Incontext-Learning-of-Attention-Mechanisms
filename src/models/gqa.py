import math
import torch
import torch.nn as nn


class GroupedQueryAttentionBlock(nn.Module):
    def __init__(
        self,
        d: int,
        hidden_dim: int,
        num_q_heads: int = 4,
        num_kv_heads: int = 2,
        causal: bool = True,
    ):
        super().__init__()
        self.d = d
        self.token_dim = d + 1
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.causal = causal

        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        if self.hidden_dim % self.num_q_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_q_heads")

        self.head_dim = self.hidden_dim // self.num_q_heads

        self.W_q = nn.Linear(self.token_dim, self.num_q_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(self.token_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(self.token_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(self.num_q_heads * self.head_dim, self.token_dim, bias=False)

        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, _ = tokens.shape

        tokens_norm = self.norm1(tokens)

        Q = self.W_q(tokens_norm)
        K = self.W_k(tokens_norm)
        V = self.W_v(tokens_norm)

        Q = Q.view(B, N, self.num_q_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        group_size = self.num_q_heads // self.num_kv_heads
        K = K.repeat_interleave(group_size, dim=1)
        V = V.repeat_interleave(group_size, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=tokens.device))
            scores = scores.masked_fill(mask.view(1, 1, N, N) == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, V)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(B, N, self.num_q_heads * self.head_dim)

        out = self.W_o(attn_out) * 0.1
        out = tokens + out
        out = self.norm2(out)
        return out


class GQATransformer(nn.Module):
    """
    Multi-layer GQA transformer for in-context linear regression.
    forward(xs, ys, query_x) -> y_pred  [B]
    """

    def __init__(
        self,
        d: int,
        num_layers: int = 1,
        hidden_dim: int | None = None,
        num_q_heads: int = 4,
        num_kv_heads: int = 2,
    ):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else d

        self.layers = nn.ModuleList(
            [
                GroupedQueryAttentionBlock(
                    d=d,
                    hidden_dim=self.hidden_dim,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.pred_head = nn.Linear(d + 1, 1, bias=False)
        nn.init.xavier_uniform_(self.pred_head.weight, gain=0.1)

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        B, n, d = xs.shape

        train_tokens = torch.cat([xs, ys.unsqueeze(-1)], dim=-1)
        query_y = torch.zeros(B, 1, device=xs.device)
        query_token = torch.cat([query_x, query_y.unsqueeze(-1)], dim=-1)
        tokens = torch.cat([train_tokens, query_token], dim=1)

        for layer in self.layers:
            tokens = layer(tokens)

        query_out = tokens[:, -1, :]
        y_pred = self.pred_head(query_out)
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
