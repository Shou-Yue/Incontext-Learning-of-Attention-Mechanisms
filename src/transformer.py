# transformer.py
"""Flexible Transformer model with switchable attention.

- use_gla=True  -> GatedLinearAttention
- use_gla=False -> MultiHeadAttention (use_softmax=False)

This switch is REQUIRED so the GD baseline (Transformer_gd) can keep using
MultiHeadAttention parameter trees produced by create_weights(), while the main
Transformer can use GLA.
"""

import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

from src.attn import (
    MLP,
    GatedLinearAttention,
    MultiHeadAttention,
    TokenVocab,
    create_pos_encoding,
    LNorm,
    layer_norm,
)


@dataclasses.dataclass
class Transformer(hk.Module):
  """Transformer with selectable attention backend."""

  def __init__(
      self,
      num_heads: int = 2,
      widening_factor: int = 4,
      num_layers: int = 3,
      key_size: int = 5,
      embedding_size: int = 64,
      output_size: int = 1,
      in_context_length: int = 17,
      in_context_length_test: int = 17,
      test_points: int = 1,
      dropout_rate: float = 0,
      only_attention: bool = True,
      use_layer_norm: bool = True,
      use_pe: bool = True,
      pe_size: int = 6,
      concat_pe: bool = False,
      output_mapping: bool = False,
      input_mapping: bool = False,
      use_bias_p: bool = True,
      zero_embeddings: bool = False,
      deq: bool = True,
      init_scale: float = 0.02,
      y_update: bool = False,
      input_mlp: bool = False,
      input_mlp_out_dim: int = 0,
      gd_mlp_config: bool = False,
      sum_norm: bool = False,
      dampening: float = 1.0,
      clip: float = 0.0,
      ana_copy: bool = False,
      flip: bool = False,
      vocab_size: int = 0,
      vocab_token_dim: int = 0,
      vocab_init: float = 0.01,
      return_logits: bool = False,
      include_query: bool = False,
      use_gla: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    self.num_heads = num_heads
    self.widening_factor = widening_factor
    self.num_layers = num_layers
    self.key_size = key_size
    self.embedding_size = embedding_size
    self.output_size = output_size

    self.in_context_length = in_context_length
    self.in_context_length_test = in_context_length_test
    self.test_points = test_points

    self.dropout_rate = dropout_rate
    self.only_attention = only_attention
    self.use_layer_norm = use_layer_norm

    self.use_pe = use_pe
    self.pe_size = pe_size
    self.concat_pe = concat_pe

    self.output_mapping = output_mapping
    self.input_mapping = input_mapping
    self.use_bias_p = use_bias_p
    self.zero_embeddings = zero_embeddings

    self.deq = deq
    self.init_scale = init_scale
    self.y_update = y_update

    self.input_mlp = input_mlp
    self.input_mlp_out_dim = input_mlp_out_dim
    self.gd_mlp_config = gd_mlp_config

    self.sum_norm = sum_norm
    self.dampening = dampening
    self.clip = clip
    self.ana_copy = ana_copy

    self.vocab_size = vocab_size
    self.vocab_token_dim = vocab_token_dim
    self.vocab_init = vocab_init
    self.return_logits = return_logits

    self.include_query = include_query
    self.use_gla = use_gla

    if pe_size > 0:
      self.pos_encoding = create_pos_encoding(in_context_length, pe_size, flip)
      self.pos_encoding_test = create_pos_encoding(in_context_length_test, pe_size, flip)
    else:
      self.pos_encoding = None
      self.pos_encoding_test = None

  def _make_attn(self, model_size: int, nl: int, *, deq_block: bool):
    """Construct an attention block with parameter naming compatible with create_weights()."""
    if self.use_gla:
      # GLA names (not used by analytic GD)
      name = "gated_linear_attention" if deq_block else ("layer_" + str(nl))
      return GatedLinearAttention(
          num_heads=self.num_heads,
          key_size=self.key_size,
          model_size=model_size,
          w_init=self.w_init,
          use_bias_p=self.use_bias_p,
          sum_normalization=self.sum_norm,
          name=name,
      )

    # MultiHeadAttention names MUST match create_weights() expectations
    # DEQ: Transformer_gd/multi_head_attention/{query,key,value,linear}
    # non-DEQ: Transformer_gd/~trans_block/layer_k/{query,key,value,linear}
    name = "multi_head_attention" if deq_block else ("layer_" + str(nl))
    return MultiHeadAttention(
        num_heads=self.num_heads,
        key_size=self.key_size,
        model_size=model_size,
        w_init=self.w_init,
        use_bias_p=self.use_bias_p,
        use_softmax=False,
        name=name,
    )

  def trans_block(self, h, nl):
    """Single block update: normalization -> attention -> residual -> optional MLP -> residual."""
    if self.deq:
      h_norm = self.lnorm1(h) if self.use_layer_norm else h
      if not self.include_query:
        key = h_norm[:, :-1, :]
        value = h_norm[:, :-1, :]
      else:
        key = h_norm
        value = h_norm
      h_attn, att_map = self.attn_block(h_norm, key, value)
    else:
      if nl == 0:
        h_norm = h
      else:
        h_norm = layer_norm(h, name="norm_" + str(nl)) if self.use_layer_norm else h

      attn_block = self._make_attn(self.model_size, nl, deq_block=False)

      if not self.include_query:
        key = h_norm[:, :-1, :]
        value = h_norm[:, :-1, :]
      else:
        key = h_norm
        value = h_norm

      h_attn, att_map = attn_block(h_norm, key, value)

    h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)

    if self.y_update:
      h = h.at[:, :, -1].set(h[:, :, -1] + self.dampening * h_attn[:, :, -1])
    else:
      h = h + self.dampening * h_attn

    if self.clip > 0:
      h = jnp.clip(h, -self.clip, self.clip)

    if not self.only_attention:
      if self.deq:
        h_inter_norm = self.lnorm2(h) if self.use_layer_norm else h
        h_dense = self.dense_block(h_inter_norm)
      else:
        h_inter_norm = layer_norm(h) if self.use_layer_norm else h
        dense_block = MLP(
            w_init=self.w_init,
            widening_factor=self.widening_factor,
            use_bias_p=self.use_bias_p,
        )
        h_dense = dense_block(h_inter_norm)

      h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
      h = h + self.dampening * h_dense

      if self.clip > 0:
        h = jnp.clip(h, -self.clip, self.clip)

    return h, att_map

  def __call__(self, x: jnp.ndarray, is_training: bool, predict_test: bool):
    # Optional vocab embedding
    if self.vocab_size > 0 and self.vocab_token_dim > 0:
      w_init_vocab = hk.initializers.VarianceScaling(self.vocab_init)
      vocab = TokenVocab(
          w_init=w_init_vocab,
          e_size=self.vocab_token_dim,
          vocab_size=self.vocab_size,
      )
      x = vocab(x)

    self.w_init = hk.initializers.VarianceScaling(self.init_scale)
    self.dropout_rate = self.dropout_rate if is_training else 0.0

    # Input mapping
    if self.input_mapping:
      embeddings = hk.Linear(
          self.embedding_size,
          with_bias=self.use_bias_p,
          w_init=self.w_init,
          name="emb",
      )(x)
    else:
      embeddings = x

    # Optional input MLP
    if self.input_mlp:
      input_mlp = MLP(
          w_init=self.w_init,
          widening_factor=self.widening_factor,
          second_layer=False,
          use_bias_p=True,
          outputdim=self.input_mlp_out_dim,
          name="input_mlp",
      )
      embeddings = embeddings + input_mlp(embeddings)

    # Positional encoding
    if self.use_pe and self.pos_encoding is not None:
      if self.concat_pe:
        pos = self.pos_encoding_test if predict_test else self.pos_encoding
        pos = pos[None, ...]
        pos = jnp.repeat(pos, embeddings.shape[0], axis=0)
        if self.zero_embeddings:
          pos = pos * 0
        h = jnp.concatenate([embeddings, pos], axis=2)
      else:
        pos = self.pos_encoding_test if predict_test else self.pos_encoding
        h = embeddings + pos
    else:
      h = embeddings

    # Infer model size
    if h.ndim == 2:
      _, model_size = h.shape
    else:
      _, _, model_size = h.shape
    self.model_size = model_size

    # DEQ setup (single reused block)
    if self.deq:
      self.attn_block = self._make_attn(model_size, nl=0, deq_block=True)
      if not self.only_attention:
        self.dense_block = MLP(
            w_init=self.w_init,
            widening_factor=self.widening_factor,
            use_bias_p=self.use_bias_p,
        )
      if self.use_layer_norm:
        self.lnorm1 = LNorm()
        self.lnorm2 = LNorm()

    st = h[:, -1, -1] * (-1.0) if not self.ana_copy else (h if self.include_query else h[:, :-1, :])
    stack_h = [] if not self.input_mlp else [st]
    stack_att = []

    for nl in range(self.num_layers):
      h, att_map = self.trans_block(h, nl)
      st = h[:, -1, -1] * (-1.0) if not self.ana_copy else (h if self.include_query else h[:, :-1, :])
      stack_h.append(st)
      stack_att.append(att_map)

    out = hk.Linear(self.output_size)(h) if self.output_mapping else h

    if self.return_logits and self.vocab_size > 0 and self.vocab_token_dim > 0:
      out = vocab(out, logits=True)

    return out, stack_h, stack_att
