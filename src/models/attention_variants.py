"""
Attention variants for in-context learning.

Includes:
  - Standard softmax attention (scaled dot-product)
  - Low-rank softmax attention (Linformer-style projection of K, V)
  - Kernelized linear attention (feature map, no softmax)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseAttentionBlock(nn.Module):
    """Common scaffolding for attention blocks."""

    def __init__(self, d, hidden_dim=None):
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        self.token_dim = d + 1

        self.W_q = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_k = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.W_v = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.W_o = nn.Linear(self.token_dim, self.token_dim, bias=False)

        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    def _attention(self, Q, K, V):
        raise NotImplementedError

    def forward(self, tokens):
        # tokens: (batch, n_tokens, token_dim)
        tokens_norm = self.norm1(tokens)

        Q = self.W_q(tokens_norm)  # (B, N, H)
        K = self.W_k(tokens_norm)  # (B, N, H)
        V = self.W_v(tokens_norm)  # (B, N, token_dim)

        attn_out = self._attention(Q, K, V)  # (B, N, token_dim)
        out = self.W_o(attn_out) * 0.1
        out = tokens + out
        out = self.norm2(out)
        return out


class SoftmaxSelfAttentionBlock(_BaseAttentionBlock):
    """Standard scaled dot-product softmax attention."""

    def __init__(self, d, hidden_dim=None, causal=False):
        super().__init__(d, hidden_dim)
        self.causal = causal

    def _attention(self, Q, K, V):
        # Q, K: (B, N, H), V: (B, N, token_dim)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        if self.causal:
            n_tokens = scores.size(-1)
            mask = torch.tril(torch.ones(n_tokens, n_tokens, device=scores.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, V)


class LowRankSoftmaxAttentionBlock(_BaseAttentionBlock):
    """
    Linformer-style low-rank approximation.

    Projects K and V along the sequence dimension (N -> k),
    then applies softmax attention in the reduced space.
    """

    def __init__(self, d, n_tokens, proj_k=None, hidden_dim=None, normalize_qk=True,
                 share_ef=False, orth_init=False, identity_proj=False, freeze_proj=False,
                 learnable_scale=True, block_size=None):
        super().__init__(d, hidden_dim)
        self.n_tokens = n_tokens
        self.block_size = block_size
        if block_size is not None:
            if n_tokens % block_size != 0:
                raise ValueError(f"block_size {block_size} must divide n_tokens {n_tokens}")
            self.num_blocks = n_tokens // block_size
            block_k = proj_k if proj_k is not None else max(1, block_size // 2)
            self.block_k = block_k
            self.proj_k = self.num_blocks * block_k
        else:
            self.proj_k = proj_k if proj_k is not None else max(1, n_tokens // 2)
        self.normalize_qk = normalize_qk
        self.share_ef = share_ef
        self.orth_init = orth_init
        self.identity_proj = identity_proj
        self.freeze_proj = freeze_proj
        self.learnable_scale = learnable_scale

        # Learnable temperature to better match softmax sharpness
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=learnable_scale)

        # Linear projections along sequence length dimension
        if self.block_size is not None:
            self.E_block = nn.Linear(self.block_size, self.block_k, bias=False)
            self.F_block = self.E_block if share_ef else nn.Linear(self.block_size, self.block_k, bias=False)
            self.E = None
            self.F = None
        else:
            self.E = nn.Linear(n_tokens, self.proj_k, bias=False)
            self.F = self.E if share_ef else nn.Linear(n_tokens, self.proj_k, bias=False)
            self.E_block = None
            self.F_block = None

        if self.identity_proj:
            if self.proj_k != self.n_tokens:
                raise ValueError("identity_proj requires proj_k == n_tokens")
            with torch.no_grad():
                self.E.weight.copy_(torch.eye(self.n_tokens, device=self.E.weight.device))
                if not self.share_ef:
                    self.F.weight.copy_(torch.eye(self.n_tokens, device=self.F.weight.device))
        elif self.orth_init:
            if self.block_size is not None:
                nn.init.orthogonal_(self.E_block.weight)
                if not self.share_ef:
                    nn.init.orthogonal_(self.F_block.weight)
            else:
                nn.init.orthogonal_(self.E.weight)
                if not self.share_ef:
                    nn.init.orthogonal_(self.F.weight)
        else:
            if self.block_size is not None:
                nn.init.xavier_uniform_(self.E_block.weight, gain=0.01)
                if not self.share_ef:
                    nn.init.xavier_uniform_(self.F_block.weight, gain=0.01)
            else:
                nn.init.xavier_uniform_(self.E.weight, gain=0.01)
                if not self.share_ef:
                    nn.init.xavier_uniform_(self.F.weight, gain=0.01)

        if self.freeze_proj:
            if self.block_size is not None:
                self.E_block.weight.requires_grad = False
                if not self.share_ef:
                    self.F_block.weight.requires_grad = False
            else:
                self.E.weight.requires_grad = False
                if not self.share_ef:
                    self.F.weight.requires_grad = False

    def _attention(self, Q, K, V):
        # Q: (B, N, H)
        if self.normalize_qk:
            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)

        # Project K and V along sequence dimension (N -> k)
        if self.block_size is not None:
            # Blockwise projection to preserve local token structure
            B, N, H = K.shape
            _, _, Dv = V.shape
            K_blocks = K.view(B, self.num_blocks, self.block_size, H)
            V_blocks = V.view(B, self.num_blocks, self.block_size, Dv)

            # (B, num_blocks, block_k, H)
            K_proj_blocks = torch.einsum('bnsh,ks->bnkh', K_blocks, self.E_block.weight)
            V_proj_blocks = torch.einsum('bnsv,ks->bnkv', V_blocks, self.F_block.weight)

            K_proj = K_proj_blocks.reshape(B, self.num_blocks * self.block_k, H)
            V_proj = V_proj_blocks.reshape(B, self.num_blocks * self.block_k, Dv)
        else:
            K_proj = self.E(K.transpose(1, 2)).transpose(1, 2)  # (B, k, H)
            V_proj = self.F(V.transpose(1, 2)).transpose(1, 2)  # (B, k, token_dim)

        scores = torch.bmm(Q, K_proj.transpose(1, 2))  # (B, N, k)
        scale = F.softplus(self.scale) if self.learnable_scale else 1.0
        scores = scores * (scale / math.sqrt(self.hidden_dim))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, V_proj)  # (B, N, token_dim)


class KernelLinearAttentionBlock(_BaseAttentionBlock):
    """
    Kernelized linear attention (feature map phi, no softmax).

    Uses phi(x) = elu(x) + 1 as in linear transformer literature.
    """

    def __init__(self, d, hidden_dim=None, eps=1e-6, normalize_qk=True):
        super().__init__(d, hidden_dim)
        self.eps = eps
        self.normalize_qk = normalize_qk
        self.scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def _attention(self, Q, K, V):
        # Feature map
        if self.normalize_qk:
            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)

        scale = F.softplus(self.scale)
        Q_phi = self._phi(Q) * scale  # (B, N, H)
        K_phi = self._phi(K)  # (B, N, H)

        # Compute K^T V once
        KV = torch.bmm(K_phi.transpose(1, 2), V)  # (B, H, token_dim)

        # Normalizer: Q_phi * sum(K_phi)
        K_sum = K_phi.sum(dim=1)  # (B, H)
        denom = torch.bmm(Q_phi, K_sum.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # Compute output
        out = torch.bmm(Q_phi, KV)  # (B, N, token_dim)
        out = out / (denom.unsqueeze(-1) + self.eps)
        return out


class MultiLayerAttentionModel(nn.Module):
    """
    Generic multi-layer attention model for in-context learning.

    Supports softmax, low-rank softmax (Linformer-style), and kernelized linear attention.
    """

    def __init__(self, d, num_layers=1, hidden_dim=None, attn_type="softmax",
                 n_tokens=None, proj_k=None, causal=False, lowrank_kwargs=None):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        self.attn_type = attn_type
        lowrank_kwargs = lowrank_kwargs or {}

        if attn_type == "softmax":
            block_fn = lambda: SoftmaxSelfAttentionBlock(d, self.hidden_dim, causal=causal)
        elif attn_type == "linformer":
            if n_tokens is None:
                raise ValueError("n_tokens is required for linformer attention")
            block_fn = lambda: LowRankSoftmaxAttentionBlock(
                d,
                n_tokens,
                proj_k=proj_k,
                hidden_dim=self.hidden_dim,
                **lowrank_kwargs
            )
        elif attn_type == "kernel":
            block_fn = lambda: KernelLinearAttentionBlock(d, self.hidden_dim)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.layers = nn.ModuleList([block_fn() for _ in range(num_layers)])

        self.pred_head = nn.Linear(d + 1, 1, bias=False)
        nn.init.xavier_uniform_(self.pred_head.weight, gain=0.1)

    def forward(self, xs, ys, query_x):
        batch_size, n_points, d = xs.shape

        train_tokens = torch.cat([xs, ys.unsqueeze(-1)], dim=-1)  # (B, n_points, d+1)
        query_y = torch.zeros(batch_size, 1, device=xs.device)
        query_token = torch.cat([query_x, query_y.unsqueeze(-1)], dim=-1)  # (B, 1, d+1)

        tokens = torch.cat([train_tokens, query_token], dim=1)  # (B, n_points+1, d+1)

        for layer in self.layers:
            tokens = layer(tokens)

        query_token_out = tokens[:, -1, :]
        y_pred = self.pred_head(query_token_out)
        return y_pred.squeeze(-1)

    def get_weight_update(self, xs, ys, query_x):
        batch_size, n_points, d = xs.shape

        with torch.no_grad():
            n_test = min(d * 2, 100)
            test_xs = torch.randn(batch_size, n_test, d, device=xs.device)

            test_preds = []
            for i in range(n_test):
                x_i = test_xs[:, i:i+1, :]
                y_pred = self.forward(xs, ys, x_i)
                test_preds.append(y_pred)

            test_preds = torch.stack(test_preds, dim=1)  # (B, n_test)

            delta_w = torch.zeros(batch_size, d, device=xs.device)
            for b in range(batch_size):
                X = test_xs[b]  # (n_test, d)
                y = test_preds[b]
                try:
                    w = torch.linalg.lstsq(X, y).solution
                    delta_w[b] = w
                except Exception:
                    w = torch.linalg.pinv(X) @ y
                    delta_w[b] = w

        return delta_w
