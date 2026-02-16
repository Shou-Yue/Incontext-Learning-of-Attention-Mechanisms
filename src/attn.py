# =======================
# src/attn.py (UPDATED: GLA now returns an attention map)
# =======================

import math
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


# -----------------------
# LayerNorm helpers
# -----------------------

class LNorm(hk.Module):
  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


# -----------------------
# Positional Encoding
# -----------------------

def create_pos_encoding(length: int, dim: int, flip: bool = False) -> jnp.ndarray:
  """Sinusoidal positional encoding."""
  position = jnp.arange(length)[:, None]
  div_term = jnp.exp(jnp.arange(0, dim, 2) * -(math.log(10000.0) / dim))

  pe = jnp.zeros((length, dim))
  pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
  pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

  if flip:
    pe = jnp.flip(pe, axis=0)
  return pe


# -----------------------
# Token embedding helper
# -----------------------

class TokenVocab(hk.Module):
  def __init__(self, vocab_size: int, embed_dim: int, name: Optional[str] = None):
    super().__init__(name=name)
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    emb = hk.Embed(vocab_size=self.vocab_size, embed_dim=self.embed_dim)
    return emb(x)


# -----------------------
# MLP block
# -----------------------

class MLP(hk.Module):
  def __init__(self, widening_factor: int, name: Optional[str] = None):
    super().__init__(name=name)
    self.widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hidden = hk.Linear(x.shape[-1] * self.widening_factor)(x)
    hidden = jax.nn.gelu(hidden)
    out = hk.Linear(x.shape[-1])(hidden)
    return out


# -----------------------
# Standard MultiHeadAttention
# -----------------------

class MultiHeadAttention(hk.Module):
  def __init__(
      self,
      num_heads: int,
      key_size: int,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      w_init: Optional[hk.initializers.Initializer] = None,
      use_bias_p: bool = False,
      # compatibility with transformer.py
      use_softmax: bool = True,
      use_non_lin_mix: bool = False,
      sum_normalization: bool = False,
      name: Optional[str] = None,
      **kwargs,
  ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or (num_heads * self.value_size)
    self.w_init = w_init
    self.use_bias_p = use_bias_p

    # stored for compatibility/debug; not used in this baseline attention
    self.use_softmax = use_softmax
    self.use_non_lin_mix = use_non_lin_mix
    self.sum_normalization = sum_normalization
    self.extra_kwargs = kwargs

  @hk.transparent
  def _proj(self, x: jnp.ndarray, head_size: int, name: str) -> jnp.ndarray:
    y = hk.Linear(
        self.num_heads * head_size,
        with_bias=self.use_bias_p,
        w_init=self.w_init,
        name=name,  # "query" / "key" / "value"
    )(x)
    *ld, T, _ = x.shape
    return y.reshape((*ld, T, self.num_heads, head_size))

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
  ):
    q = self._proj(query, self.key_size, "query")
    k = self._proj(key, self.key_size, "key")
    v = self._proj(value, self.value_size, "value")

    scale = 1.0 / math.sqrt(self.key_size)
    attn_logits = jnp.einsum("...thd,...shd->...ths", q, k) * scale

    if mask is not None:
      attn_logits = attn_logits + mask

    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    out = jnp.einsum("...ths,...shd->...thd", attn_weights, v)

    *ld, T, H, D = out.shape
    out = out.reshape((*ld, T, H * D))

    out = hk.Linear(
        self.model_size,
        with_bias=self.use_bias_p,
        w_init=self.w_init,
        name="linear",
    )(out)

    return out, attn_weights


# -----------------------
# Gated Linear Attention
# -----------------------

class GatedLinearAttention(hk.Module):
  """Feature-map linear attention + a multiplicative gate.

  Returns an 'effective attention' map for visualization:
    att_map[..., h, tq, tk] = soft row-normalized phi(q_t)^T phi(k_s)
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      w_init: Optional[hk.initializers.Initializer] = None,
      use_bias_p: bool = False,
      # compatibility with transformer.py
      sum_normalization: bool = False,
      name: Optional[str] = None,
      **kwargs,
  ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or (num_heads * self.value_size)
    self.w_init = w_init
    self.use_bias_p = use_bias_p

    # stored for compatibility/debug; not used here
    self.sum_normalization = sum_normalization
    self.extra_kwargs = kwargs

  @hk.transparent
  def _proj(self, x: jnp.ndarray, head_size: int, name: str) -> jnp.ndarray:
    y = hk.Linear(
        self.num_heads * head_size,
        with_bias=self.use_bias_p,
        w_init=self.w_init,
        name=name,  # "query" / "key" / "value"
    )(x)
    *ld, T, _ = x.shape
    return y.reshape((*ld, T, self.num_heads, head_size))

  @hk.transparent
  def _phi(self, x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.elu(x) + 1.0

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,  # ignored (non-causal here)
  ):
    # projections
    q = self._proj(query, self.key_size, "query")          # [..., Tq, H, K]
    k = self._proj(key,   self.key_size, "key")            # [..., Tk, H, K]
    v = self._proj(value, self.value_size, "value")        # [..., Tk, H, V]

    # feature map
    qf = self._phi(q)                                      # [..., Tq, H, K]
    kf = self._phi(k)                                      # [..., Tk, H, K]

    # ----- effective attention map for visualization -----
    # scores: [..., H, Tq, Tk]
    scores = jnp.einsum("...thk,...shk->...hts", qf, kf)
    # row-normalize over keys (Tk)
    att_map = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-6)

    # ----- linear attention output -----
    # S = Σ_t kf_t ⊗ v_t  => [..., H, K, V]
    S = jnp.einsum("...thk,...thv->...hkv", kf, v)

    # denom = qf · Σ_t kf_t  => [..., Tq, H]
    k_sum = jnp.sum(kf, axis=-3)  # [..., H, K]
    denom = jnp.einsum("...thk,...hk->...th", qf, k_sum) + 1e-6

    # out = (qf @ S) / denom  => [..., Tq, H, V]
    out = jnp.einsum("...thk,...hkv->...thv", qf, S) / denom[..., None]

    # Gate MUST be computed from query (Tq), not value (Tk)
    gate = hk.Linear(
        self.num_heads * self.value_size,
        with_bias=True,
        w_init=hk.initializers.Constant(0.0),
        b_init=hk.initializers.Constant(0.0),
        name="gla_gate",
    )(query)

    *ld, Tq, _ = query.shape
    gate = gate.reshape((*ld, Tq, self.num_heads, self.value_size))
    gate = jax.nn.sigmoid(gate)  # ~0.5 at init

    # scale by 2 so effective initial multiplier ≈ 1
    out = out * (2.0 * gate)

    # merge heads
    *ld, Tq2, H, D = out.shape
    out = out.reshape((*ld, Tq2, H * D))

    # output projection name matches MHA downstream expectations
    out = hk.Linear(
        self.model_size,
        with_bias=self.use_bias_p,
        w_init=self.w_init,
        name="linear",
    )(out)

    return out, att_map
