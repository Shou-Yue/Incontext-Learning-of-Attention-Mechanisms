# Constructed Token GLA Notebook (JAX/Haiku)

## From Linear Attention to Gated Linear Attention (GLA)

## Overview

This project modifies the original paper's implementation by replacing
**Linear Attention** with **Gated Linear Attention (GLA)** inside the
Transformer architecture.

The goal is to study how introducing gating into the attention mechanism
changes optimization dynamics, stability, expressivity, and alignment
with Gradient Descent (GD) in an in-context linear regression setting.

The notebook:

-   Implements constructed-token in-context regression
-   Replaces linear attention with gated linear attention
-   Trains the modified Transformer using JAX + Haiku + Optax
-   Compares results against explicit Gradient Descent baselines
-   Measures alignment between Transformer updates and GD
-   Analyzes attention behavior and stability

------------------------------------------------------------------------

## What Changed: Linear Attention → Gated Linear Attention

### Original (Linear Attention)

The paper's implementation uses linear attention of the form:

    Attention(Q, K, V) ≈ φ(Q) (φ(K)^T V)

This removes the quadratic softmax attention cost and produces a
linear-time attention mechanism.

However, it lacks a mechanism to dynamically modulate information flow
between tokens.

### Modified (Gated Linear Attention)

We introduce a gating mechanism:

    Output = Gate ⊙ LinearAttention(Q, K, V)

Where:

-   Gate = σ(W_g · input)
-   σ is a sigmoid (or similar) activation
-   ⊙ is elementwise multiplication

This allows the model to: - Suppress irrelevant context contributions -
Dynamically scale update magnitude - Improve stability in recurrent /
DEQ-style settings

The gating mechanism is implemented in `attn.py` and integrated into the
Transformer stack in `transformer.py`.

------------------------------------------------------------------------

## Repository Structure

Recommended layout:

project_root/ Constructed_token_GLA.ipynb src/ **init**.py attn.py \#
Gated Linear Attention implementation transformer.py \# Transformer
using GLA train.py data.py config.py

------------------------------------------------------------------------

## Environment Setup

Required packages:

-   Python 3.9+
-   jax
-   jaxlib
-   dm-haiku
-   optax
-   ml_collections
-   numpy
-   matplotlib
-   pillow

Install dependencies:

pip install -U dm-haiku optax ml_collections

If JAX errors occur, ensure jax and jaxlib versions are compatible with
your hardware (CPU/GPU).

------------------------------------------------------------------------

## Experiment Setup

### Task: In-Context Linear Regression

Each training example consists of:

-   Context: (x₁, y₁), ..., (xₙ, yₙ)
-   Query: x\*
-   Target: y\*

The model must infer y\* using only context tokens.

### Constructed Tokens

Each token is concatenated:

-   Context token: \[x_i \|\| y_i\]
-   Query token: \[x\_\* \|\| 0\]

The Transformer predicts the missing y component.

------------------------------------------------------------------------

## Training Process

Each training iteration:

1.  Sample a batch of regression tasks
2.  Construct tokens
3.  Forward pass using Gated Linear Attention
4.  Compute MSE loss on query prediction
5.  Backpropagate and update parameters

Periodic evaluation includes:

-   Transformer test loss
-   GD baseline loss
-   GD++ tuned baseline loss
-   Cosine similarity alignment metrics
-   Functional interpolation between Transformer and GD

------------------------------------------------------------------------

## Why Gated Linear Attention?

The switch from linear attention to GLA allows us to test:

1.  Does gating improve stability in recurrent (DEQ-style) Transformers?
2.  Does gating improve alignment with Gradient Descent updates?
3.  Does gating change attention sparsity or structure?
4.  Does gating improve convergence speed or final loss?

------------------------------------------------------------------------

## Interpreting Results

### Loss Curves

If GLA improves optimization, expect:

-   Faster convergence
-   Lower final test loss
-   More stable training dynamics

### Cosine Similarity (Alignment with GD)

Higher cosine similarity indicates that the Transformer's learned
updates align more closely with true gradient descent steps.

If GLA improves alignment, this supports the hypothesis that gating
helps approximate optimizer-like dynamics.

### Attention Patterns

With gating, attention weights may:

-   Become more selective
-   Suppress noisy context points
-   Show structured query-to-context flow

------------------------------------------------------------------------

## Expected Observations

Compared to linear attention, GLA may:

-   Improve conditioning of updates
-   Reduce instability in recurrent settings
-   Increase robustness to larger dataset_size
-   Better approximate iterative optimization dynamics

------------------------------------------------------------------------

## Summary

This project re-implements the paper's linear attention architecture
using Gated Linear Attention to study how gating affects in-context
learning and optimizer-like behavior.

The key experimental comparison is:

Linear Attention vs Gated Linear Attention

measured through:

-   Test loss
-   GD alignment
-   Interpolation behavior
-   Attention structure
