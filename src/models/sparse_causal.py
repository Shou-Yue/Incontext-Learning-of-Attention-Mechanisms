import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def build_sparse_causal_mask(
    seq_len: int,
    window_size: int,
    stride: Optional[int] = None,
    global_tokens: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    i = torch.arange(seq_len, device=device).view(seq_len, 1)
    j = torch.arange(seq_len, device=device).view(1, seq_len)

    causal = j <= i
    local = (i - j) < window_size
    allowed = causal & local

    if stride is not None and stride > 0:
        strided = ((i - j) % stride == 0) & causal
        allowed = allowed | strided

    if global_tokens > 0:
        allowed = allowed | ((j < global_tokens) & causal)

    return allowed


class SparseCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        window_size: int,
        stride: Optional[int] = None,
        global_tokens: int = 0,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.window_size = window_size
        self.stride = stride
        self.global_tokens = global_tokens

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        allowed = build_sparse_causal_mask(
            seq_len=t,
            window_size=self.window_size,
            stride=self.stride,
            global_tokens=self.global_tokens,
            device=x.device,
        )

        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        att = att.masked_fill(~allowed, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y


class SparseSelfAttentionBlock(nn.Module):
    """
    Sparse causal attention block over token_dim = d+1 (same layout as other ICL models).
    """
    def __init__(
        self,
        d: int,
        hidden_dim: int | None = None,
        window_size: int = 128,
        stride: Optional[int] = None,
        global_tokens: int = 0,
    ):
        super().__init__()
        self.token_dim = d + 1
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        self.window_size = window_size
        self.stride = stride
        self.global_tokens = global_tokens

        self.W_q = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_k = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_v = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.W_o = nn.Linear(self.token_dim, self.token_dim, bias=False)

        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)

        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, d+1]
        tokens_norm = self.norm1(tokens)
        Q = self.W_q(tokens_norm)
        K = self.W_k(tokens_norm)
        V = self.W_v(tokens_norm)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        allowed = build_sparse_causal_mask(
            seq_len=scores.size(-1),
            window_size=self.window_size,
            stride=self.stride,
            global_tokens=self.global_tokens,
            device=tokens.device,
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.bmm(weights, V)

        out = self.W_o(attn_out) * 0.1
        out = tokens + out
        out = self.norm2(out)
        return out


class MLP(nn.Module):
    def __init__(self, n_embd: int, mlp_ratio: int = 4, resid_pdrop: float = 0.0):
        super().__init__()
        hidden = mlp_ratio * n_embd
        self.fc1 = nn.Linear(n_embd, hidden)
        self.fc2 = nn.Linear(hidden, n_embd)
        self.drop = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        window_size: int,
        stride: Optional[int] = None,
        global_tokens: int = 0,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SparseCausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            window_size=window_size,
            stride=stride,
            global_tokens=global_tokens,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, mlp_ratio=mlp_ratio, resid_pdrop=resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SparseGPTBackbone(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_embd: int,
        n_head: int,
        window_size: int,
        stride: Optional[int] = None,
        global_tokens: int = 0,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    window_size=window_size,
                    stride=stride,
                    global_tokens=global_tokens,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x


class SparseTransformerModel(nn.Module):
    """
    Mirrors the TransformerModel API/behavior, but uses sparse causal attention.
    """
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int = 64,
        n_layer: int = 12,
        n_head: int = 4,
        window_size: int = 128,
        stride: Optional[int] = None,
        global_tokens: int = 0,
        resid_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
    ):
        super().__init__()

        self.name = (
            f"sparse_gpt_embd={n_embd}_layer={n_layer}_head={n_head}"
            f"_win={window_size}_stride={stride}_global={global_tokens}"
        )

        self.n_positions = n_positions
        self.n_dims = n_dims

        self.read_in = nn.Linear(n_dims, n_embd)
        # Positional + type embeddings are critical for interleaved x/y tokens.
        # Without them, the model cannot distinguish x vs y positions.
        self.pos_emb = nn.Embedding(n_positions, n_embd)
        self.type_emb = nn.Embedding(2, n_embd)
        self.backbone = SparseGPTBackbone(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            window_size=window_size,
            stride=stride,
            global_tokens=global_tokens,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b: torch.Tensor, ys_b: torch.Tensor) -> torch.Tensor:
        bsize, points, dim = xs_b.shape

        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            dim=2,
        )

        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        zs = self._combine(xs, ys)
        embeds = self.read_in(zs)
        # Add positional + type embeddings (even= x token, odd= y token)
        t = embeds.size(1)
        pos_ids = torch.arange(t, device=embeds.device)
        if t > self.pos_emb.num_embeddings:
            pos_ids = pos_ids % self.pos_emb.num_embeddings
        type_ids = (pos_ids % 2).long()
        embeds = embeds + self.pos_emb(pos_ids).unsqueeze(0)
        embeds = embeds + self.type_emb(type_ids).unsqueeze(0)
        h = self.backbone(embeds)
        prediction = self.read_out(h)
        # Predict at y positions (odd indices) where model has seen the x token
        prediction = prediction[:, 1::2, 0]
        return prediction


class SparseICLModel(nn.Module):
    """
    Wrapper to match the common ICL API:
      forward(xs, ys, query_x) -> y_pred [B]
    """
    def __init__(
        self,
        d: int,
        n_points: Optional[int] = None,
        num_layers: int = 1,
        hidden_dim: int | None = None,
        n_head: int = 4,
        window_size: int = 128,
        stride: Optional[int] = None,
        global_tokens: int = 0,
    ):
        super().__init__()
        self.d = d
        self.n_points = n_points
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else d

        # Use the same tokenization as other ICL models (x,y concatenated).
        self.layers = nn.ModuleList([
            SparseSelfAttentionBlock(
                d=d,
                hidden_dim=self.hidden_dim,
                window_size=window_size,
                stride=stride,
                global_tokens=global_tokens,
            )
            for _ in range(num_layers)
        ])

        self.pred_head = nn.Linear(d + 1, 1, bias=False)
        nn.init.xavier_uniform_(self.pred_head.weight, gain=0.1)

    def forward(self, xs, ys, query_x):
        b, n, d = xs.shape
        train_tokens = torch.cat([xs, ys.unsqueeze(-1)], dim=-1)  # [B, n, d+1]
        query_y = torch.zeros(b, 1, device=xs.device, dtype=xs.dtype)
        query_token = torch.cat([query_x, query_y.unsqueeze(-1)], dim=-1)  # [B, 1, d+1]
        tokens = torch.cat([train_tokens, query_token], dim=1)  # [B, n+1, d+1]

        for layer in self.layers:
            tokens = layer(tokens)

        query_out = tokens[:, -1, :]  # [B, d+1]
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
