# Constructed Token GLA Notebook (JAX/Haiku)

## Overview

This project trains a Transformer model on an in-context linear
regression task using a constructed-token representation. It compares
learned behavior against explicit Gradient Descent (GD) baselines and
analyzes whether the Transformer implicitly learns optimizer-like
dynamics.

The notebook: - Builds synthetic regression tasks - Trains a Transformer
using JAX + Haiku + Optax - Compares results against explicit gradient
descent - Measures alignment between Transformer updates and GD -
Visualizes attention patterns and learned weights

------------------------------------------------------------------------

## Repository Structure

Recommended layout:

project_root/ Constructed_token_GLA.ipynb src/ **init**.py attn.py
transformer.py train.py data.py config.py

If the files are not under `src/`, modify imports inside the notebook
accordingly.

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

Install missing dependencies:

pip install -U dm-haiku optax ml_collections

If you encounter JAX version mismatch errors, install a compatible pair
of jax and jaxlib for your hardware (CPU/GPU).

------------------------------------------------------------------------

## Experiment Description

### Task

Each training example is a linear regression task:

-   Context: (x₁, y₁), ..., (xₙ, yₙ)
-   Query: x\*
-   Target: y\*

The model must predict y\* using only the in-context examples.

### Constructed Tokens

Each token concatenates inputs and labels:

-   Context token: \[x_i \|\| y_i\]
-   Query token: \[x\_\* \|\| 0\]

The Transformer predicts the missing y component.

### Model

The Transformer can be configured as: - Standard multi-layer attention
model - Recurrent / weight-shared (DEQ-style) model

Key configuration parameters (in config.py): - dataset_size (number of
context points) - input_size (dimension of x) - key_size (model width) -
num_layers, num_heads - training_steps - batch size

------------------------------------------------------------------------

## Training Process

Each training step:

1.  Sample batch of regression tasks
2.  Construct tokens
3.  Forward pass through Transformer
4.  Compute MSE loss on query prediction
5.  Backpropagate and update parameters

Periodic evaluation includes: - Transformer test loss - GD baseline
loss - GD++ tuned baseline loss - Cosine similarity alignment metrics -
Functional interpolation between models

------------------------------------------------------------------------

## Interpreting Results

### Loss Curves

-   Decreasing Transformer loss indicates successful learning.
-   Matching GD++ loss suggests optimizer-like behavior.

### Cosine Similarity

Measures alignment between Transformer updates and GD updates.
Increasing similarity supports the interpretation that the model
implements gradient-like updates internally.

### Interpolation

If interpolating between Transformer and GD predictions yields low loss
at intermediate mixing values, their behaviors are related but not
identical.

### Attention Patterns

Structured query-to-context attention suggests algorithmic behavior
rather than memorization.

------------------------------------------------------------------------

## Common Issues

-   ModuleNotFoundError: ensure files are inside src/ or adjust imports.
-   JAX errors: install matching jax + jaxlib versions.
-   Shape mismatches: restart kernel and run notebook top-to-bottom
    after modifying configuration values.

------------------------------------------------------------------------

## Summary

This project studies whether Transformers trained on synthetic
regression tasks learn implicit gradient descent behavior when using
constructed tokens. Results should be interpreted using both loss
metrics and alignment diagnostics rather than loss alone.
