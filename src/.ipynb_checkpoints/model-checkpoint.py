import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers import GPT2Config, GPT2Model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd = 256, n_layer = 12, n_head = 8):
        super().__init__()
        config = GPT2Config(
            n_positions = 2 * n_positions,
            n_embd = n_embd,
            n_layer = n_layer,
            n_head = n_head,
            resid_pdrop = 0.0,
            embd_pdrop = 0.0,
            attn_pdrop = 0.0,
            use_cache = False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims

        self.read_in = nn.Linear(n_dims, n_embd)
        self.backbone = GPT2Model(config)
        self.read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleaves x's and y's into a single sequence

        Args:
            xs_b: [B, T, D] input vectors
            ys_b: [B, T] corresponding scalar outputs

        Returns:
            zs: [B, 2T, D]  where positions are x_1, y_1, x_2, y_2, ...
        """
        bsize, points, dim = xs_b.shape

        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device = ys_b.device),
            ),
            dim = 2,
        )

        zs = torch.stack((xs_b, ys_b_wide), dim = 2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys):
        """
        Returns predictions for all y positions

        Args:
            xs: [B, T, D] input vectors
            ys: [B, T] ground-truth outputs

        Returns:
            pred: [B, T]
        """
        zs = self._combine(xs, ys)
        embeds = self.read_in(zs)
        output = self.backbone(inputs_embeds = embeds).last_hidden_state
        prediction = self.read_out(output)
        prediction = prediction[:, ::2, 0]
        return prediction


def build_sparse_causal_mask(
    seq_len: int,
    window_size: int,
    stride: Optional[int] = None,
    global_tokens: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Boolean mask [T, T] where True means attention is allowed.

    Pattern (all causal):
      - local window: attend to previous `window_size` tokens
      - optional strided: also attend to positions where (i - j) % stride == 0
      - optional global: always attend to first `global_tokens` positions
    """
    i = torch.arange(seq_len, device = device).view(seq_len, 1)
    j = torch.arange(seq_len, device = device).view(1, seq_len)

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

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias = True)
        self.proj = nn.Linear(n_embd, n_embd, bias = True)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T, C]
        """
        b, t, c = x.shape

        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim = 2)

        # [B, nh, T, hd]
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        allowed = build_sparse_causal_mask(
            seq_len = t,
            window_size = self.window_size,
            stride = self.stride,
            global_tokens = self.global_tokens,
            device = x.device,
        )  # [T, T] bool

        # Manual scaled dot-product attention (compatible with PyTorch < 2.0)
        scale = 1.0 / (self.head_dim ** 0.5)

        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, nh, T, T]

        # allowed: True = keep, so block where ~allowed is True
        att = att.masked_fill(~allowed, float("-inf"))

        att = torch.softmax(att, dim = -1)
        att = self.attn_drop(att)

        y = torch.matmul(att, v)  # [B, nh, T, hd]

        y = y.transpose(1, 2).contiguous().view(b, t, c)  # [B, T, C]
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
            n_embd = n_embd,
            n_head = n_head,
            window_size = window_size,
            stride = stride,
            global_tokens = global_tokens,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd = n_embd, mlp_ratio = mlp_ratio, resid_pdrop = resid_pdrop)

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
                    n_embd = n_embd,
                    n_head = n_head,
                    window_size = window_size,
                    stride = stride,
                    global_tokens = global_tokens,
                    attn_pdrop = attn_pdrop,
                    resid_pdrop = resid_pdrop,
                    mlp_ratio = mlp_ratio,
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
    Mirrors your TransformerModel API/behavior, but uses sparse causal attention.
    """
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int = 256,
        n_layer: int = 12,
        n_head: int = 8,
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
            n_layer = n_layer,
            n_embd = n_embd,
            n_head = n_head,
            window_size = window_size,
            stride = stride,
            global_tokens = global_tokens,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop,
        )
        self.read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b: torch.Tensor, ys_b: torch.Tensor) -> torch.Tensor:
        bsize, points, dim = xs_b.shape

        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device = ys_b.device),
            ),
            dim = 2,
        )

        zs = torch.stack((xs_b, ys_b_wide), dim = 2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        zs = self._combine(xs, ys)           # [B, 2T, D]
        embeds = self.read_in(zs)            # [B, 2T, E]
        h = self.backbone(embeds)            # [B, 2T, E]
        prediction = self.read_out(h)        # [B, 2T, 1]
        prediction = prediction[:, ::2, 0]   # [B, T]
        return prediction
