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
        h = self.backbone(embeds)
        prediction = self.read_out(h)
        prediction = prediction[:, ::2, 0]
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
        n_embd = hidden_dim if hidden_dim is not None else d
        n_positions = (2 * n_points + 2) if n_points is not None else (2 * d + 1)
        self.backbone = SparseTransformerModel(
            n_dims=d,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=num_layers,
            n_head=n_head,
            window_size=window_size,
            stride=stride,
            global_tokens=global_tokens,
        )

    def forward(self, xs, ys, query_x):
        b, n, d = xs.shape
        query_y = torch.zeros(b, 1, device=xs.device, dtype=xs.dtype)
        xs_all = torch.cat([xs, query_x], dim=1)
        ys_all = torch.cat([ys, query_y], dim=1)
        preds = self.backbone(xs_all, ys_all)  # [B, n+1]
        return preds[:, -1]

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
