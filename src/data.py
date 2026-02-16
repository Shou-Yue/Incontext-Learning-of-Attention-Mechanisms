# data.py
"""Data utils.
  Provide functions to create regression datasets.
"""

from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data(rng, i_size, c_size, size_distract, input_range, w_scale):
  """Create a linear regression data set: X*w where x ~ U(-1, 1), w ~ N(0,1)."""

  rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
  w = jax.random.normal(rng, shape=[i_size])*w_scale

  x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                         minval=-input_range/2, maxval=input_range/2)
  x_querry = jax.random.uniform(new_rng2, shape=[1, i_size],
                                minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.squeeze(x@w)
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)
  y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                   shape=[size_distract]))

  y_target = x_querry@w
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data[..., None]], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  x_querry_init = -1*x_querry.dot(jnp.ones_like(x_querry).T*0.0)
  zero = jnp.concatenate([x_querry, x_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), w

data_creator = vmap(create_reg_data,
                    in_axes=(0, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=1), 3, 10, 0, 2, 1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_sin(rng, i_size, c_size, size_distract, input_range, w_scale):
  """Create a non-linear regression data set: sin(x*w)."""

  rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
  w = jax.random.normal(rng, shape=[i_size])*w_scale

  x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                         minval=-input_range/2, maxval=input_range/2)
  x_querry = jax.random.uniform(new_rng2, shape=[1, i_size],
                                minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.squeeze(jnp.sin(x@w))
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)
  y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                   shape=[size_distract]))

  y_target = jnp.sin(x_querry@w)
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data[..., None]], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  x_querry_init = -1*x_querry.dot(jnp.ones_like(x_querry).T*0.0)
  zero = jnp.concatenate([x_querry, x_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), w


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_sin_test(rng, i_size, c_size, size_distract, input_range):
  """Create a non-linear regression test data set: sin(x*w)."""

  rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
  w = jax.random.normal(rng, shape=[i_size])

  x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                         minval=-input_range/2, maxval=input_range/2)
  x_querry = jax.random.uniform(new_rng2, shape=[1, i_size],
                                minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.squeeze(jnp.sin(x@w))
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)
  y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                   shape=[size_distract]))

  y_target = jnp.sin(x_querry@w)
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data[..., None]], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  x_querry_init = -1*x_querry.dot(jnp.ones_like(x_querry).T*0.0)
  zero = jnp.concatenate([x_querry, x_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), w


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_classic_token(rng, i_size, c_size, size_distract, input_range, w_scale):
  """Create a data set with classic token construction."""

  rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
  w = jax.random.normal(rng, shape=[i_size])*w_scale

  x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                         minval=-input_range/2, maxval=input_range/2)
  x_querry = jax.random.uniform(new_rng2, shape=[1, i_size],
                                minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.squeeze(x@w)
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)
  y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                   shape=[size_distract]))

  y_target = x_querry@w
  y_target = y_target[..., None]

  x_tokens = jnp.concatenate([x, x_querry], 0)
  y_tokens = jnp.concatenate([y_data[..., None], y_target], 0)

  seq = []
  for i in range(c_size):
    seq.append(jnp.concatenate([x_tokens[i], jnp.array([0.0])], -1))
    seq.append(jnp.concatenate([jnp.zeros_like(x_tokens[i]), y_tokens[i]], -1))
  seq.append(jnp.concatenate([x_tokens[-1], jnp.array([0.0])], -1))
  seq.append(jnp.concatenate([jnp.zeros_like(x_tokens[-1]), y_tokens[-1]], -1))

  seq = jnp.stack(seq, axis=0)
  target = jnp.concatenate([x_querry, y_target], -1)
  return jnp.squeeze(seq), jnp.squeeze(target), w


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_ood_data(rng, i_size, c_size, input_range, w_scale):
  """Create an out-of-distribution regression dataset by scaling inputs."""

  rng, new_rng, new_rng2 = jax.random.split(rng, 3)
  w = jax.random.normal(rng, shape=[i_size])*w_scale

  x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                         minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.squeeze(x@w)
  x_querry = jax.random.uniform(new_rng2, shape=[1, i_size],
                                minval=-input_range/2, maxval=input_range/2)
  y_target = x_querry@w
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data[..., None]], -1)
  x_querry_init = -1*x_querry.dot(jnp.ones_like(x_querry).T*0.0)
  zero = jnp.concatenate([x_querry, x_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)

  target = jnp.concatenate([x_querry, y_target], -1)
  return jnp.squeeze(seq), jnp.squeeze(target), w


def create_weights(
    i_size: int,
    o_size: int,
    c_size: int,
    lr: float,
    w_init: jnp.ndarray,
    second_zero: bool = False,
    lin_diag: bool = False,
    gd_deq: bool = False,
    num_layers: int = 1,
    input_mlp_rnd: jax.Array | None = None,
    in_proj: bool = False,
):
  """Create linear-regression gradient-descent weights for Transformer_gd.

  Updated to match newer Haiku module paths that include:
    Transformer_gd/~_make_attn/multi_head_attention/{query,key,value,linear}

  It also keeps backward-compatible keys for older layouts, by writing aliases.
  """
  dim = i_size + o_size
  one = jnp.ones([dim])
  one_in = jnp.ones([i_size])
  one_out = jnp.ones([o_size])
  zero_out = jnp.zeros([o_size])

  # -----------------------
  # Value projection matrix
  # -----------------------
  value_upper = jnp.zeros([i_size, dim])          # [i_size, dim]
  value_lower_left = w_init[0]                    # [o_size, i_size]
  value_lower_right = jnp.diag(one_out) * (-2.0 if lin_diag else -1.0)
  if second_zero:
    value_lower_right = jnp.diag(zero_out)

  value_lower = jnp.concatenate([value_lower_left, value_lower_right], axis=1)  # [o_size, dim]
  value_matrix = jnp.concatenate([value_upper, value_lower], axis=0).T          # [dim, dim]
  if lin_diag:
    value_matrix = value_matrix + jnp.diag(one)

  # -----------------------
  # Query / Key matrices
  # -----------------------
  query_upper = jnp.zeros([o_size, dim])          # [o_size, dim]
  query_lower_left = jnp.diag(one_in)             # [i_size, i_size]
  query_lower_right = jnp.zeros([i_size, o_size]) # [i_size, o_size]
  query_lower = jnp.concatenate([query_lower_left, query_lower_right], axis=1)  # [i_size, dim]
  query_matrix = jnp.concatenate([query_lower, query_upper], axis=0)            # [dim, dim]
  key_matrix = query_matrix

  # -----------------------
  # Output projection matrix
  # -----------------------
  proj_upper = jnp.zeros([i_size, dim])           # [i_size, dim]
  proj_lower_left = jnp.zeros([o_size, i_size])   # [o_size, i_size]
  proj_lower_right = jnp.diag(one_out) * ((1.0 / c_size) * lr)  # [o_size, o_size]

  if lin_diag:
    proj_lower_right = proj_lower_right + (jnp.diag(one_out) * (1.0 / c_size) * (1.0 / c_size) * lr)

  proj_lower = jnp.concatenate([proj_lower_left, proj_lower_right], axis=1)     # [o_size, dim]
  projection_matrix = jnp.concatenate([proj_upper, proj_lower], axis=0)         # [dim, dim]

  if lin_diag:
    projection_matrix = projection_matrix - jnp.diag(one) * (1.0 / c_size) * (1.0 / c_size) * lr

  # -----------------------
  # Pack Haiku params dict
  # -----------------------
  params_new: dict[str, dict[str, jnp.ndarray]] = {}

  def _put(base_prefix: str):
    """base_prefix must end with '/'."""
    params_new[base_prefix + "query"]  = {"w": jnp.array(query_matrix)}
    params_new[base_prefix + "key"]    = {"w": jnp.array(key_matrix)}
    params_new[base_prefix + "value"]  = {"w": jnp.array(value_matrix)}
    params_new[base_prefix + "linear"] = {"w": jnp.array(projection_matrix)}

  for l in range(num_layers):
    # In some runs the GD transformer constructs attention under "~_make_attn/multi_head_attention/"
    # (this is the exact path in your error).
    if num_layers == 1 or gd_deq:
      # Write BOTH: old and new layouts.
      _put("Transformer_gd/multi_head_attention/")
      _put("Transformer_gd/~_make_attn/multi_head_attention/")
    else:
      # Per-layer layout: write a small family of likely prefixes.
      # Your codebase has used variants across experiments; keeping aliases avoids breakage.
      _put(f"Transformer_gd/~trans_block/layer_{l}/multi_head_attention/")
      _put(f"Transformer_gd/~trans_block/layer_{l}/~_make_attn/multi_head_attention/")
      _put(f"Transformer_gd/~trans_block/layer_{l}/")  # legacy (no multi_head_attention segment)

  # Optional input projection (Transformer_gd uses hk.Linear(..., name="emb") when input_mapping=True)
  if in_proj:
    if input_mlp_rnd is None:
      raise ValueError("in_proj=True requires input_mlp_rnd to be provided.")
    rng1, _, _ = jax.random.split(input_mlp_rnd, 3)
    w_emb = jax.random.normal(rng1, shape=[dim, dim]) * jnp.sqrt(0.002 / dim)
    params_new["Transformer_gd/emb"] = {"w": w_emb}

  # Optional input MLP path (only used if your config enables it)
  elif input_mlp_rnd is not None:
    rng1, rng2, _ = jax.random.split(input_mlp_rnd, 3)
    # These shapes match your existing branch.
    w1 = jax.random.normal(rng1, shape=[40, 160]) * jnp.sqrt(0.002 / 2.0)
    w2 = jax.random.normal(rng2, shape=[160, 40]) * jnp.sqrt(0.002 / 40.0)
    b1 = jnp.zeros([160])
    b2 = jnp.zeros([40])
    w_emb = jax.random.normal(rng1, shape=[2, 40]) * jnp.sqrt(0.002 / 2.0)

    params_new["Transformer_gd/input_mlp/linear"] = {"w": w1, "b": b1}
    params_new["Transformer_gd/input_mlp/linear_1"] = {"w": w2, "b": b2}
    params_new["Transformer_gd/emb"] = {"w": w_emb}

  return params_new
