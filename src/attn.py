# =======================
# src/attn.py (UPDATED: GLA is causal + VALUE-gated + returns an attention map)
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
# Gated Linear Attention (CAUSAL + VALUE-GATED)
# -----------------------

class GatedLinearAttention(hk.Module):
  """Causal feature-map linear attention with multiplicative VALUE gate.

  Output (causal linear attention):
    kf = phi(k), qf = phi(q)
    gate_t = sigmoid(W_g x_t) in R^{H*V} (computed from value input)
    v_t' = v_t * (2 * gate_t)  # init multiplier ~= 1
    Kcum_t  = Σ_{i<=t} kf_i
    KVcum_t = Σ_{i<=t} kf_i ⊗ v_i'
    out_t = (qf_t @ KVcum_t) / (qf_t · Kcum_t + eps)

  Also returns a visualization attention map:
    att_map[..., h, t, s] ∝ qf[...,t,h,:] · kf[...,s,h,:] for s<=t, row-normalized.
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
      mask: Optional[jnp.ndarray] = None,  # not used for computation; causal is enforced structurally
  ):
    # projections
    q = self._proj(query, self.key_size, "query")          # [..., Tq, H, K]
    k = self._proj(key,   self.key_size, "key")            # [..., Tk, H, K]
    v = self._proj(value, self.value_size, "value")        # [..., Tk, H, V]

    *ld, Tq, _ = query.shape
    *ld2, Tk, _ = key.shape

    if Tq < Tk:
      raise ValueError(f"GLA causal self-attn expects Tq >= Tk, got Tq={Tq}, Tk={Tk}")

    # feature map
    qf = self._phi(q)                                      # [..., Tq, H, K]
    kf = self._phi(k)                                      # [..., Tk, H, K]

    # ----- VALUE gating (computed from 'value' input, i.e. tokens at key positions) -----
    gate = hk.Linear(
        self.num_heads * self.value_size,
        with_bias=True,
        w_init=hk.initializers.Constant(0.0),
        b_init=hk.initializers.Constant(0.0),
        name="gla_gate",
    )(value)                                               # [..., Tk, H*V]
    gate = gate.reshape((*ld2, Tk, self.num_heads, self.value_size))
    gate = jax.nn.sigmoid(gate)                            # init ~0.5
    v = v * (2.0 * gate)                                   # init multiplier ~1

    # ----- effective attention map for visualization (O(T^2), causal-triangular) -----
    # scores: [..., H, Tq, Tk]
    scores = jnp.einsum("...thk,...shk->...hts", qf, kf)    # dot in feature space
    # apply causal mask by zeroing future keys (supports rectangular Tq x Tk)
    causal = (jnp.arange(Tk)[None, :] <= jnp.arange(Tq)[:, None]).astype(scores.dtype)  # [Tq, Tk]
    scores = scores * causal[None, None, :, :]
    att_map = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-6)

    # ----- causal linear attention via prefix sums (O(T)) -----
    # KV term per key position: [..., Tk, H, K, V]
    kv = jnp.einsum("...thk,...thv->...thkv", kf, v)
    kv_cum = jnp.cumsum(kv, axis=-4)                       # cumulative over Tk
    k_cum = jnp.cumsum(kf, axis=-3)                        # [..., Tk, H, K]

    # gather prefix sums for each query position t using idx[t] = min(t, Tk-1)
    idx = jnp.minimum(jnp.arange(Tq), Tk - 1)              # [Tq]
    kv_sel = jnp.take(kv_cum, idx, axis=-4)                # [..., Tq, H, K, V]
    k_sel = jnp.take(k_cum, idx, axis=-3)                  # [..., Tq, H, K]

    # numerator: qf_t @ KVsel_t  => [..., Tq, H, V]
    numer = jnp.einsum("...thk,...thkv->...thv", qf, kv_sel)

    # denom: qf_t · Ksel_t       => [..., Tq, H]
    denom = jnp.einsum("...thk,...thk->...th", qf, k_sel) + 1e-6

    out = numer / denom[..., None]                         # [..., Tq, H, V]

    # merge heads
    *ld3, T, H, D = out.shape
    out = out.reshape((*ld3, T, H * D))

    # output projection name matches MHA downstream expectations
    out = hk.Linear(
        self.model_size,
        with_bias=self.use_bias_p,
        w_init=self.w_init,
        name="linear",
    )(out)

    return out, att_map