# Design Document: Linear and Low-Rank Attention Mechanisms for In-Context Learning

**Project:** Evaluating Different Attention Mechanisms for In-Context Learning as Optimization Methods  
**Focus Area:** Linear Attention & Low-Rank Attention Approximations  
**Authors:** Krish Prasad (and team)  
**Date:** January 2026  
**Status:** Quarter 2 Implementation Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background and Motivation](#background-and-motivation)
3. [Research Questions](#research-questions)
4. [System Architecture](#system-architecture)
5. [Attention Mechanism Specifications](#attention-mechanism-specifications)
6. [Experimental Design](#experimental-design)
7. [Evaluation Framework](#evaluation-framework)
8. [Implementation Plan](#implementation-plan)
9. [Data Pipeline](#data-pipeline)
10. [Technical Requirements](#technical-requirements)
11. [Expected Outcomes](#expected-outcomes)
12. [Risks and Mitigations](#risks-and-mitigations)

---

## Executive Summary

### What We're Building

This project investigates whether **optimization-like in-context learning** (ICL)—specifically, the ability to internally perform gradient descent—observed in standard transformers with **softmax attention** is preserved when using efficiency-constrained attention approximations. We focus on two variants:

1. **Linear Attention** (O(n) complexity in sequence length, kernel-based approximation)
2. **Low-Rank Attention** (rank-r approximation of softmax attention)

### Why It Matters

Prior research demonstrates two key findings:
1. **Theoretical**: Linear self-attention (LSA, no softmax) is mathematically equivalent to gradient descent (von Oswald et al., 2023)
2. **Empirical**: Standard transformers with **softmax attention** exhibit gradient descent-like ICL on linear regression tasks (Garg et al., 2023; our Q1 work)

**Key Gap:** While softmax attention is the baseline used in production systems, we don't know if efficiency approximations (linear attention, low-rank) preserve this optimization-like ICL behavior.

### What We'll Learn

By systematically comparing baseline (softmax), linear, and low-rank attention on **identical** linear regression tasks, we'll determine:

- Which architectural properties are essential for GD-like ICL
- How efficiency constraints affect in-context learning quality
- Whether approximations preserve the optimization behavior
- Trade-offs between computational efficiency and learning capability

---

## Background and Motivation

### The In-Context Learning Phenomenon

Transformers can adapt to new tasks at inference time by conditioning on examples in the prompt, without updating parameters. For example, given:

```
Prompt: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ), x_query
Task: Predict y_query where yᵢ = w*ᵀxᵢ for some unknown w*
```

A trained transformer can output accurate predictions as if it learned w* from the examples.

### The Optimization Hypothesis

Recent theoretical and empirical work suggests transformers achieve ICL by **implementing optimization algorithms internally**. Specifically:

- Linear self-attention can be proven mathematically equivalent to one step of gradient descent
- Trained transformers match the predictions of iterative learners (least squares, LASSO, etc.)
- Models exhibit gradient-like update directions when analyzed

### Quarter 1 Findings

Our team established:

1. **LSA ≈ Gradient Descent:** Single-layer linear self-attention predictions align with one-step GD on MSE loss
2. **Transformer ≈ Specialized Learners:** Multi-layer models match least squares (underparameterized), min-norm LS (overparameterized), and LASSO (sparse)
3. **Pretrained LLMs Show GD Patterns:** GPT models exhibit high cosine similarity with GD-1 baseline on linear regression prompts

### The Critical Gap

While we know that:
- **LSA** (no softmax) has a theoretical proof of equivalence to GD
- **Softmax attention** (standard transformers) shows strong empirical GD-like behavior

We need controlled experiments on efficiency approximations:
- **Linear attention** (kernel-based, O(n) complexity) - removes softmax normalization
- **Low-rank attention** (rank-r factorization) - constrains attention rank
- **Other practical variants** (studied by other team members)

to determine if optimization-like ICL requires specific properties of softmax attention or generalizes to approximations.

---

## Research Questions

### Primary Question

**To what extent do linear and low-rank attention mechanisms support gradient descent-like in-context learning compared to standard softmax attention?**

### Sub-Questions

1. **Performance:** Do linear/low-rank attention achieve comparable in-context prediction error on linear regression tasks?

2. **Optimization Alignment:** Do these mechanisms produce updates that align with gradient descent directions (measured via cosine similarity)?

3. **Robustness:** How do these mechanisms handle:
   - Varying prompt lengths (n in-context examples)
   - Different problem dimensions (d)
   - Label noise (ϵ ~ N(0, σ²))
   - Input distribution shifts (e.g., skewed covariance)

4. **Efficiency Trade-offs:** What is the relationship between:
   - Computational cost (time/memory scaling)
   - Approximation quality (rank r, kernel choice)
   - ICL performance (error, GD alignment)

5. **Architectural Necessity:** Which attention properties are **essential** for GD-like ICL:
   - Softmax normalization?
   - Full-rank attention matrix?
   - Specific parameterization (QKV decomposition)?

---

## System Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Experimental Pipeline                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Generation Module                     │
│  • Synthetic linear regression task sampler                  │
│  • Prompt constructor: (x₁,y₁,...,xₙ,yₙ,x_query)            │
│  • Distribution controls: P_X, P_w, noise level              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Model Architecture Module                    │
│                                                               │
│  ┌─────────────────────────────────────────────────┐        │
│  │   Transformer Backbone (Fixed)                  │        │
│  │   • Embedding: R^d → R^d_model                  │        │
│  │   • Positional encoding                          │        │
│  │   • N layers of (Attention + MLP)               │        │
│  │   • Output projection: R^d_model → R            │        │
│  └─────────────────────────────────────────────────┘        │
│                       │                                       │
│                       ▼                                       │
│  ┌─────────────────────────────────────────────────┐        │
│  │   Attention Module (VARIABLE - Swappable)       │        │
│  │                                                  │        │
│  │   [Baseline]  Standard Softmax Attention        │        │
│  │   [Linear]    Kernel-based Linear Attention     │        │
│  │   [Low-Rank]  Rank-r Approximation              │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Training Module                           │
│  • Loss: MSE(ŷ_query, y_query)                              │
│  • Optimizer: AdamW (fixed hyperparameters)                  │
│  • Curriculum: dimension warmup (5d → d)                     │
│  • Logging: loss, metrics, attention patterns                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Evaluation Module                           │
│                                                               │
│  [Metric 1] In-Context Prediction Error                      │
│  • MSE on held-out prompts vs varying n                      │
│                                                               │
│  [Metric 2] Cosine Similarity to Gradient Descent            │
│  • Extract implicit weight update: Δw_model                  │
│  • Compare to GD-1: Δw_GD = (1/n)X^T(y - Xw₀)               │
│  • Compute: cos(Δw_model, Δw_GD)                             │
│                                                               │
│  [Metric 3] Robustness Analysis                              │
│  • Noise injection: y ← y + ϵ, ϵ ~ N(0,σ²)                  │
│  • Scaling: dimension d, prompt length n, input scale α      │
│  • Distribution shift: P_X (isotropic → skewed)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Analysis & Visualization                     │
│  • Learning curves (all mechanisms overlaid)                 │
│  • Cosine similarity plots vs n                              │
│  • Attention pattern heatmaps                                │
│  • Scaling analysis (time/memory vs performance)             │
└─────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Single Variable Isolation:** Only the attention mechanism changes; all other components (data, optimizer, architecture depth/width) remain fixed.

2. **Shared Interface:** All attention modules expose the same API:
   ```python
   def forward(Q, K, V, mask=None) -> output
   # Input:  Q, K, V ∈ R^(batch × n × d_head)
   # Output: output ∈ R^(batch × n × d_head)
   ```

3. **Drop-in Replacement:** Configuration flag controls mechanism:
   ```bash
   python train.py --attn_type {baseline | linear | lowrank}
   ```

4. **Reproducibility:** Fixed random seeds, versioned datasets, logged hyperparameters.

---

## Attention Mechanism Specifications

### Baseline: Standard Softmax Attention

**Purpose:** Control condition representing real-world transformer attention.

**Computation:**
```
A = softmax(QK^T / √d_k)
Output = AV
```

**Complexity:**
- Time: O(n² · d)
- Memory: O(n²) for attention matrix A

**Properties:**
- Full-rank attention (rank ≤ n)
- Non-linear due to softmax
- Normalized attention weights (rows sum to 1)

---

### Mechanism 1: Linear Attention

**Purpose:** Test if O(n) complexity mechanisms preserve GD-like ICL.

#### What is Linear Attention?

Linear attention (Katharopoulos et al., 2020) uses **kernel feature maps** to replace softmax, reducing complexity from O(n²) to O(n) by exploiting the **associativity of matrix multiplication**.

**Key Idea:**  
Express self-attention as a linear dot-product of kernel feature maps. By reordering operations, compute φ(K)^T V once and reuse for all queries, avoiding the n×n attention matrix entirely.

#### Mathematical Formulation

Standard attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Linear attention with kernel feature map φ:
```
V'_i = [φ(Q_i)^T ∑_{j=1}^n φ(K_j) V_j^T] / [φ(Q_i)^T ∑_{j=1}^n φ(K_j)]

Vectorized: V' = φ(Q) [φ(K)^T V] / [φ(Q) [φ(K)^T 1]]
                       └─────┘ ← Precompute once: O(n·d²)
```

Where:
- φ: R^d_k → R^d_φ is a feature map ensuring non-negative similarity
- Associativity: φ(Q)[φ(K)^T V] instead of [φ(Q)φ(K)^T]V eliminates the O(n²) term

#### Implementation Details

**Feature Map Choice:**

We use **φ(x) = elu(x) + 1** following Katharopoulos et al. (2020):
- Ensures non-negativity: sim(q,k) = φ(q)^T φ(k) ≥ 0 (required for valid attention)
- Smooth gradients (unlike ReLU which zeros out negative values)
- Empirically performs on par with softmax on image generation and ASR
- Dimension: C = d_k (no dimension explosion)

**Implementation:** Standard multi-head attention with φ applied to Q, K before the associative computation φ(Q)[φ(K)^T V]. See implementation reference doc for details.

**Complexity (Katharopoulos et al., 2020):**
- Time: O(n · d² · d_v) for computing φ(K)^T V, then O(n · d²) for applying φ(Q)
  - Total: O(n · max(d², d·d_v)) → **linear in n**
- Memory: O(d · d_v) for cached φ(K)^T V matrix
  - **No O(n²) attention matrix stored**

**Properties:**
- Approximates softmax when φ chosen appropriately (kernel trick)
- Enables **constant-time autoregressive inference** (RNN formulation)
- Can be written as recurrent network with states s_i = s_{i-1} + φ(K_i)V_i^T

#### Why This Tests Our Hypothesis

If linear attention achieves comparable GD-like ICL:
- **Conclusion:** Softmax normalization is not essential; kernel approximation suffices
- **Implication:** ICL may emerge from attention's information-routing capability, not specific normalization

If linear attention fails:
- **Conclusion:** Softmax's normalization or expressiveness is critical
- **Implication:** Full pairwise attention may be necessary for optimization-like behavior

---

### Mechanism 2: Low-Rank Attention

**Purpose:** Test if low-rank approximations (computational efficiency via reduced rank) preserve GD-like ICL.

#### What is Low-Rank Attention?

Low-rank attention (Linformer - Wang et al., 2020) **projects** keys and values to a lower dimension k << n before computing attention, based on the empirical observation that self-attention matrices are **low-rank**.

**Key Idea:**  
The self-attention matrix P ∈ R^(n×n) can be approximated by a low-rank matrix with rank k = O(log n). Instead of computing full QK^T, project K, V down to dimension k first.

#### Mathematical Formulation

**Linformer Approach (Wang et al., 2020):**

Project K, V from sequence length n → k using learned projections E, F ∈ R^(n×k):

```
K_low = E^T K    where E ∈ R^(n × k), K_low ∈ R^(k × d_k)
V_low = F^T V    where F ∈ R^(n × k), V_low ∈ R^(k × d_v)

A = softmax(Q K_low^T / √d_k)    [n × k attention matrix]
Output = A V_low
```

**Complexity:**
- Time: O(n · k · d_k + n · k · d_v) → **O(n·k) when k << n**
- Memory: O(n · k) for attention matrix (vs O(n²) for baseline)
- Wang et al. show k = O(log n) sufficient theoretically; k ∈ {128, 256} works empirically

**Alternative: Query/Key Low-Rank Factorization (our implementation):**

Project Q, K into rank-r subspace (different from Linformer's sequence-length projection):

```
Q_r = Q W_Q^r    where W_Q^r ∈ R^(d_k × r)
K_r = K W_K^r    where W_K^r ∈ R^(d_k × r)

A = softmax(Q_r K_r^T / √r)    [still n × n, but lower-rank]
Output = A V
```

**Complexity:**
- Time: O(n² · r + n · d_k · r) → still O(n²) but with smaller constant
- Memory: O(n²) for attention matrix (same asymptotic as baseline)
- **Note:** This differs from Linformer; we test rank constraints on attention computation itself

#### Implementation Details

**Our Implementation:** We use the query/key low-rank factorization (different from Linformer's E/F projections) for direct comparison with baseline while controlling rank.

**Why not exact Linformer?** Linformer's E, F projections are fixed-size (n×k) which requires knowing max sequence length n in advance. Our approach is more flexible for variable-length sequences in our linear regression task.

**Architecture:** Project Q, K through additional learned matrices W_Q^r, W_K^r ∈ R^{d_k × r} before computing attention scores. This creates a rank-constrained attention computation. See implementation reference doc for details.

**Rank Selection:**
- Motivated by Wang et al.'s finding that self-attention is low-rank
- Start with r = d_k / 2  (e.g., if d_k = 32, try r = 16)
- Sweep: r ∈ {2, 4, 8, 16, 32} to find minimal rank preserving GD-like ICL
- Wang et al. show rank k = O(log n) sufficient; we test if this holds for ICL tasks

**Properties:**
- Retains softmax normalization (unlike linear attention)
- Reduces effective dimensionality of attention computation
- Smooth interpolation: r = d_k recovers baseline

#### Why This Tests Our Hypothesis

If low-rank attention achieves comparable GD-like ICL:
- **Conclusion:** Full-rank attention is not necessary; low-dimensional projections suffice
- **Implication:** Gradient descent-like behavior emerges from coarse-grained token routing

If low-rank attention degrades gracefully with r:
- **Insight:** We can quantify the minimal rank needed for optimization-like ICL
- **Practical value:** Guides efficient transformer design

If low-rank fails even at moderate r:
- **Conclusion:** Fine-grained attention detail is essential
- **Implication:** Efficiency approximations may harm ICL capability

---

## Experimental Design

### Shared Experimental Contract

All experiments across team members (baseline, linear, low-rank, sparse, gated, etc.) follow this protocol to ensure comparability.

#### Task Family: Linear Regression

**Task Definition:**
- Input dimension: d ∈ {10, 20, 30, 40, 50}
- Prompt length: n in-context examples (varying: 1 to 2d)
- Target function: y = w*^T x where w* ~ N(0, I_d)
- Input distribution: x ~ N(0, I_d)
- Optional noise: y ← y + ϵ, ϵ ~ N(0, σ²)

**Prompt Structure:**
```
Input:  (x₁, y₁, x₂, y₂, ..., xₙ, yₙ, x_query)
Output: ŷ_query
```

Where:
- Each xᵢ ∈ R^d is sampled i.i.d. from P_X
- yᵢ = w*^T xᵢ + ϵᵢ
- Model must predict ŷ_query ≈ w*^T x_query

**Train/Test Split:**
- **Training:** Generate fresh prompts each batch (32M distinct prompts over training)
- **Evaluation:** 1,280 held-out prompts per setting (same prompts across all mechanisms)

#### Fixed Hyperparameters (Across All Mechanisms)

```python
# Model architecture
d_model = 256
n_layers = 4
n_heads = 8
d_head = d_model // n_heads  # 32
d_mlp = 4 * d_model  # 1024
dropout = 0.0

# Training
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 0.01
batch_size = 64
total_steps = 100_000  # (reduced from 500k for initial experiments)
warmup_steps = 5_000

# Curriculum (dimension warmup)
start_dim = 5
target_dim = d  # {10, 20, 30, 40, 50}
curriculum_steps = 20_000
step_increment = 1 dimension every 2000 steps

# Logging
log_interval = 100 steps
eval_interval = 1000 steps
checkpoint_interval = 5000 steps
```

### Experimental Matrix

#### Experiment 1: Baseline Comparison (Core Result)

**Goal:** Compare linear and low-rank attention to baseline on standard task.

**Settings:**
- d = 20 (primary focus, matches Q1 work)
- n varies: {5, 10, 20, 40} in-context examples
- No noise: σ = 0
- Input distribution: P_X = N(0, I_20)

**Mechanisms:**
1. Baseline (softmax attention)
2. Linear (ELU+1 feature map)
3. Low-rank (r ∈ {8, 16, 32})

**Metrics:**
- In-context MSE vs n
- Cosine similarity to GD-1 vs n
- Training curves (loss vs steps)

**Deliverable:** 3-panel figure showing:
- (a) MSE curves overlaid for all mechanisms
- (b) Cosine similarity vs n
- (c) Training loss over time

---

#### Experiment 2: Scaling Analysis

**Goal:** Test how mechanisms scale with problem dimension d.

**Settings:**
- d ∈ {10, 20, 30, 40, 50}
- n = 2d (sufficient examples to recover w*)
- No noise: σ = 0

**Mechanisms:**
- Baseline
- Linear
- Low-rank (r = d/4)

**Metrics:**
- Final MSE at convergence vs d
- Cosine similarity vs d
- Training time per step vs d
- Memory usage vs d

**Deliverable:** Scaling plots showing how performance and efficiency degrade with dimension.

---

#### Experiment 3: Robustness to Noise

**Goal:** Test resilience to label noise.

**Settings:**
- d = 20
- n = 40
- Noise levels: σ² ∈ {0, 0.1, 0.5, 1.0, 2.0}
- 5 random seeds per setting

**Mechanisms:**
- Baseline
- Linear
- Low-rank (r = 16)

**Baseline comparisons:**
- Least squares (optimal under noise)
- Ridge regression (with cross-validated λ)

**Metrics:**
- MSE vs σ²
- Cosine similarity vs σ²

**Deliverable:** Noise robustness curves showing whether mechanisms degrade gracefully.

---

#### Experiment 4: Distribution Shift

**Goal:** Test generalization to out-of-distribution prompts (Q1 follow-up).

**Settings:**
- d = 20
- n = 40

**Distribution shift types:**

A. **Skewed Covariance:**
   - Train: P_X = N(0, I_d)
   - Test: P_X = N(0, Σ) where Σ has eigenvalues ∝ 1/i²

B. **Different Orthants:**
   - Train: standard prompts
   - Test: all in-context examples in one random orthant, query in another

**Mechanisms:**
- Baseline
- Linear
- Low-rank (r = 16)

**Metrics:**
- MSE on shifted distribution
- Cosine similarity on shifted distribution

**Deliverable:** Robustness comparison showing which mechanisms maintain GD-alignment under shift.

---

#### Experiment 5: Low-Rank Sweep (Mechanism 2 Deep Dive)

**Goal:** Characterize rank-performance trade-off.

**Settings:**
- d = 20
- n = 40
- Rank: r ∈ {2, 4, 8, 16, 32, 64}  (64 = full rank = baseline)

**Metrics:**
- MSE vs r
- Cosine similarity vs r
- Attention matrix rank (measured via SVD)
- Inference time vs r

**Deliverable:** Rank sweep plot showing:
- Smooth interpolation from r=2 (heavily constrained) to r=64 (baseline)
- "Elbow" point where additional rank has diminishing returns

---

#### Experiment 6: Attention Pattern Analysis

**Goal:** Visualize what each mechanism learns.

**Settings:**
- d = 20
- n = 20
- Single fixed prompt (for interpretability)

**Visualizations:**

A. **Attention Heatmaps:**
   - Plot A ∈ R^(n×n) for baseline and low-rank
   - For linear attention: plot φ(Q)φ(K)^T (unnormalized)

B. **Singular Value Spectrum:**
   - Compute SVD of learned attention matrices
   - Compare effective rank across mechanisms

C. **Update Direction Alignment:**
   - Extract Δw_model = f(in-context examples)
   - Visualize cos(Δw_model, Δw_GD) per token

**Deliverable:** Figure panel showing attention patterns and their relationship to GD updates.

---

### Implementation & Experimental Timeline

**Phase 1: Setup & Implementation**
- Implement LinearAttention and LowRankAttention modules
- Write unit tests and verify correctness
- Integrate into shared training pipeline
- Run smoke tests to ensure all mechanisms train

**Phase 2: Core Experiments**
- Baseline comparison (Experiment 1)
- Scaling analysis (Experiment 2)
- Rank sweep for low-rank attention (Experiment 5)

**Phase 3: Robustness Analysis**
- Noise robustness (Experiment 3)
- Distribution shift (Experiment 4)
- Attention pattern visualization (Experiment 6)

**Phase 4: Synthesis**
- Combine results with team members' mechanisms
- Generate all figures and analysis
- Write technical report and prepare website

---

## Evaluation Framework

### Metric 1: In-Context Prediction Error

**Definition:**
```
MSE = E_prompts [ (ŷ_query - y_query)² ]
```

**Computation:**
1. Generate 1,280 evaluation prompts
2. For each prompt, forward pass through trained model → ŷ_query
3. Compute squared error with ground truth y_query = w*^T x_query
4. Average across prompts

**Normalization:**
- Report raw MSE
- Also report normalized MSE = MSE / Var(y_query) (so 1.0 = naive mean baseline)

**Baseline comparisons:**
- Least Squares: w_LS = (X^T X)^{-1} X^T y
- Min-norm LS: w_min = X^+ y (for overparameterized)
- 3-Nearest Neighbors
- Averaging estimator

**Interpretation:**
- Lower MSE = better in-context learning
- Compare to optimal baselines to assess "how close to perfect"

---

### Metric 2: Cosine Similarity to Gradient Descent

**Goal:** Quantify whether the model's implicit update aligns with gradient descent.

**Gradient Descent Baseline (GD-1):**

For a linear regression task with MSE loss, one step of GD from initialization w₀ = 0 is:

```
∇L(w) = -(1/n) X^T (y - Xw)
Δw_GD = -η ∇L(0) = (η/n) X^T y

where η is step size (we use η = 1 for simplicity)
```

**Model's Implicit Update:**

We infer the model's learned weight vector by:

```python
def extract_implicit_weights(model, prompt):
    # prompt = (x₁, y₁, ..., xₙ, yₙ, x_query)
    # Collect model predictions on canonical basis vectors
    predictions = []
    for i in range(d):
        e_i = torch.zeros(d)
        e_i[i] = 1.0
        prompt_test = prompt + [(e_i, None)]  # Add e_i as query
        pred = model(prompt_test)
        predictions.append(pred)
    
    w_model = torch.tensor(predictions)  # ∈ R^d
    return w_model
```

Alternatively (simpler for linear models):
```python
# Directly fit w_model by solving:
# X_test @ w_model ≈ ŷ_test
# where X_test are query inputs, ŷ_test are model predictions

w_model = torch.linalg.lstsq(X_test, y_pred).solution
```

**Cosine Similarity Computation:**

```python
def cosine_similarity(w_model, w_GD):
    return torch.dot(w_model, w_GD) / (torch.norm(w_model) * torch.norm(w_GD))
```

**Reporting:**
- Compute cos(w_model, w_GD) for each evaluation prompt
- Average across prompts
- Plot vs n (number of in-context examples)

**Interpretation:**
- cos(w_model, w_GD) = 1.0: perfect alignment (model = GD)
- cos(w_model, w_GD) ≈ 0: orthogonal (no GD-like behavior)
- cos(w_model, w_GD) < 0: anti-aligned (unlikely, would indicate pathology)

**Expected Baseline Results (from Q1):**
- Linear self-attention (LSA, no softmax): ~0.99 cosine similarity
- Standard softmax attention (our baseline): 0.95-0.98 (strong alignment)
- Target: Our baseline should replicate the softmax attention results from Q1

---

### Metric 3: Robustness to Scaling and Noise

**Dimensions of Robustness:**

A. **Prompt Length (n):**
   - Vary n from 1 to 2d
   - Expect MSE to decrease as n → d (more examples)
   - Test if mechanisms maintain GD alignment across n

B. **Problem Dimension (d):**
   - Vary d ∈ {10, 20, 30, 40, 50}
   - Test if efficiency-constrained mechanisms scale better
   - Measure training time, memory, and final performance

C. **Label Noise (σ²):**
   - Inject Gaussian noise: y ← y + ϵ, ϵ ~ N(0, σ²)
   - Test if mechanisms degrade gracefully
   - Compare to optimal noisy baseline (Ridge regression)

D. **Input Scaling (α):**
   - Scale inputs: x ← α · x
   - Test sensitivity to input magnitude
   - Expected behavior: performance should be scale-invariant (or predictably degrade)

**Reporting:**
- For each dimension, plot MSE and cosine similarity
- Include error bars (90% CI from bootstrap or multiple seeds)
- Overlay baseline comparisons

---

### Metric 4: Computational Efficiency

**Time Complexity Measurement:**

```python
import time

def benchmark_attention(mechanism, seq_lengths, d_model, n_trials=10):
    times = []
    for n in seq_lengths:
        x = torch.randn(1, n, d_model).cuda()
        
        # Warmup
        _ = mechanism(x)
        
        # Measure
        start = time.time()
        for _ in range(n_trials):
            _ = mechanism(x)
        end = time.time()
        
        times.append((end - start) / n_trials)
    
    return times
```

**Memory Complexity Measurement:**

```python
import torch.cuda

def measure_memory(mechanism, n, d_model):
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1, n, d_model).cuda()
    _ = mechanism(x)
    return torch.cuda.max_memory_allocated() / 1e6  # MB
```

**Expected Complexity:**

| Mechanism | Time     | Memory  | Notes                          |
|-----------|----------|---------|--------------------------------|
| Baseline  | O(n²d)   | O(n²)   | Full attention matrix          |
| Linear    | O(nd²)   | O(d²)   | No n² term                     |
| Low-Rank  | O(n²r)   | O(n²)   | Reduced d → r, still O(n²)     |

**Trade-off Analysis:**
- Plot: Performance (MSE) vs Computational Cost (time)
- Pareto frontier: which mechanisms are not dominated?

---

## Data Pipeline

### Prompt Generation

**Training Prompts:**

```python
def generate_prompt(d, n, noise_std=0.0):
    # Sample task-specific weight vector
    w_star = torch.randn(d)
    
    # Sample in-context examples
    X = torch.randn(n, d)
    y = X @ w_star
    
    # Add noise
    if noise_std > 0:
        y += noise_std * torch.randn(n)
    
    # Sample query
    x_query = torch.randn(d)
    y_query = x_query @ w_star
    
    return {
        'X': X,              # (n, d)
        'y': y,              # (n,)
        'x_query': x_query,  # (d,)
        'y_query': y_query,  # scalar
        'w_star': w_star     # (d) - ground truth (for analysis)
    }
```

**Prompt Formatting (for Transformer Input):**

```python
def format_prompt(prompt):
    # Interleave (x, y) pairs + query
    # Shape: (2n + 1, d)
    
    n, d = prompt['X'].shape
    sequence = torch.zeros(2*n + 1, d)
    
    for i in range(n):
        sequence[2*i] = prompt['X'][i]         # xᵢ
        sequence[2*i + 1] = append_scalar(prompt['X'][i], prompt['y'][i])  # (xᵢ, yᵢ)
    
    sequence[2*n] = prompt['x_query']  # query
    
    return sequence

def append_scalar(x, y):
    # Map scalar y to same dimension as x (zero-pad)
    # x: (d,), y: scalar → (d,)
    return torch.cat([torch.tensor([y]), torch.zeros(len(x) - 1)])
```

Note: The exact formatting depends on the shared embedding scheme. Garg et al. use linear projections to map inputs/outputs into d_model-dimensional space.

**Evaluation Prompts:**

```python
# Generate fixed evaluation set
eval_prompts = []
for setting in eval_settings:
    for seed in range(1280):
        torch.manual_seed(seed)
        prompt = generate_prompt(d=setting['d'], n=setting['n'], noise_std=setting['noise'])
        eval_prompts.append(prompt)

# Save to disk for reproducibility
torch.save(eval_prompts, 'eval_prompts.pt')
```

### Curriculum Learning

**Motivation:** Training high-dimensional models from scratch is difficult. Gradually increasing task complexity speeds up convergence.

**Schedule:**

```python
def get_curriculum_dim(step, start_dim=5, target_dim=20, total_steps=20000):
    if step >= total_steps:
        return target_dim
    
    # Linear warmup
    progress = step / total_steps
    current_dim = start_dim + (target_dim - start_dim) * progress
    return int(current_dim)

# Example:
# Step 0:     dim = 5
# Step 2000:  dim = 6
# Step 4000:  dim = 7
# ...
# Step 20000: dim = 20
```

**Prompt Length Curriculum (optional):**

```python
def get_curriculum_n(step, start_n=5, target_n=40, total_steps=20000):
    if step >= total_steps:
        return target_n
    
    progress = step / total_steps
    current_n = start_n + (target_n - start_n) * progress
    return int(current_n)
```

---

## Implementation Plan

### Module Development

**LinearAttention Module:**
- Implement multi-head attention with φ(x) = elu(x) + 1 feature map
- Use efficient einsum operations: φ(Q)[φ(K)^T V] to achieve O(n) complexity
- Unit tests: shape checks, NaN detection, gradient flow verification

**LowRankAttention Module:**
- Implement rank-constrained attention via learnable projections W_Q^r, W_K^r
- Configurable rank parameter r
- Ensure r = d_k recovers baseline behavior
- Unit tests across multiple rank values

### Integration

**Configuration System:**
- Drop-in attention mechanism selection via `--attn_type` flag
- Shared hyperparameters across all mechanisms
- Consistent interface: all modules expose same forward(Q, K, V) signature

**Training Pipeline:**
- Standard transformer training loop with MSE loss
- Curriculum learning: dimension warmup (5 → d)
- Logging: loss, MSE, cosine similarity metrics
- Checkpointing at regular intervals

### Validation

**Smoke Test:**
- Small-scale experiment (d=10, n=20, 10k steps)
- Verify all mechanisms train without errors
- Check basic convergence (cosine similarity > 0.5 for baseline)
- Runtime budget: ~2 hours for all three mechanisms

**Success Criteria:**
- No NaNs or training instabilities
- Loss curves decrease monotonically
- Gradients flow properly through all attention variants

*See `IMPLEMENTATION_REFERENCE.md` for detailed code skeletons and API specifications.*

---

## Technical Requirements

### Software Dependencies

```python
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.15.0  # For experiment tracking
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
tqdm>=4.65.0
```

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB
- Storage: 50GB

**Recommended:**
- GPU: NVIDIA A100 or V100 (40GB VRAM)
- RAM: 32GB
- Storage: 100GB (for checkpoints, logs)

**Expected Runtimes (d=20, 100k steps):**
- Baseline attention: ~6 hours (A100)
- Linear attention: ~5 hours (slightly faster)
- Low-rank attention: ~5-6 hours (depends on r)

### Code Organization

```
Incontext-Learning-of-Attention-Mechanisms/
├── README.md
├── DESIGN_DOC.md  (this file)
├── WARP.md
├── requirements.txt
├── config.py
├── train.py
├── evaluate.py
├── data/
│   ├── generate_prompts.py
│   └── curriculum.py
├── models/
│   ├── transformer.py
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── baseline_attention.py
│   │   ├── linear_attention.py
│   │   └── lowrank_attention.py
│   └── mlp.py
├── evaluation/
│   ├── metrics.py
│   ├── baselines.py  (Least Squares, etc.)
│   └── visualize.py
├── experiments/
│   ├── exp1_baseline_comparison.py
│   ├── exp2_scaling.py
│   ├── exp3_noise_robustness.py
│   ├── exp4_distribution_shift.py
│   ├── exp5_lowrank_sweep.py
│   └── exp6_attention_analysis.py
├── scripts/
│   ├── run_all_experiments.sh
│   └── generate_figures.py
├── tests/
│   ├── test_linear_attention.py
│   ├── test_lowrank_attention.py
│   └── test_data_generation.py
└── results/
    ├── checkpoints/
    ├── logs/
    └── figures/
```

---

## Expected Outcomes

### Hypotheses and Predictions

#### Hypothesis 1: Linear Attention Preserves GD-like ICL

**Prediction:**
- Linear attention will achieve 80-95% of baseline's performance on standard linear regression
- Cosine similarity will remain > 0.85 (compared to baseline's ~0.95)
- Performance gap widens under distribution shift (e.g., skewed covariance)

**Rationale:**
- Kernel feature maps can approximate softmax attention
- Loss of softmax normalization may hurt, but not catastrophically
- Linear attention's O(n) scaling is beneficial for long prompts

**If confirmed:** Softmax normalization is not essential; kernel approximation suffices.

**If refuted:** Softmax's normalization is critical for optimization-like ICL.

---

#### Hypothesis 2: Low-Rank Attention Degrades Gracefully

**Prediction:**
- Smooth performance curve: as r ↑, MSE ↓
- "Elbow" around r = d_k / 4 to d_k / 2
- Cosine similarity maintained even at moderate r (16-32)

**Rationale:**
- Linear regression is relatively low-complexity
- Attention may not need full expressiveness for simple tasks
- Low-rank = information bottleneck, but not necessarily catastrophic

**If confirmed:** Efficiency approximations are viable; can use r << d_k without major loss.

**If refuted:** Full-rank attention is necessary; fine-grained token interactions matter.

---

#### Hypothesis 3: Mechanisms Differ in Robustness

**Prediction:**
- Baseline: robust to noise, sensitive to input scale
- Linear: more sensitive to noise (no softmax damping), robust to scale
- Low-rank: moderate robustness, depends on r

**Rationale:**
- Softmax normalization acts as implicit regularization
- Linear attention lacks this, may overfit to noise
- Low-rank constraints may act as regularization

**If confirmed:** Trade-offs between efficiency and robustness exist; design choice depends on task properties.

---

### Success Criteria

**Minimum Viable Results:**
1. All mechanisms train successfully (converging loss curves)
2. At least one mechanism achieves cosine similarity > 0.85 on standard task
3. Clear performance ranking: baseline vs linear vs low-rank (r=16)

**Strong Results:**
1. Linear attention achieves 90%+ of baseline performance
2. Low-rank attention with r=16 achieves 85%+ of baseline performance
3. Smooth rank-performance trade-off curve (Experiment 5)
4. Interpretable attention pattern visualizations

**Outstanding Results:**
1. Linear attention **matches** baseline on standard task
2. Low-rank attention with r=8 still achieves 80%+ performance
3. Novel insights: e.g., "linear attention is more robust to distribution shift"
4. Clear recommendations for practitioners

---

## Risks and Mitigations

### Risk 1: Implementation Bugs

**Risk:** Linear/low-rank attention modules have subtle bugs (e.g., incorrect einsum, numerical instability).

**Mitigation:**
- Comprehensive unit tests (shape checks, gradient checks, NaN detection)
- Compare against reference implementations (e.g., from papers)
- Incremental development: test each component separately before integration

---

### Risk 2: Training Instability

**Risk:** New mechanisms don't converge; loss plateaus or explodes.

**Mitigation:**
- Curriculum learning (proven to help in Q1)
- Gradient clipping (norm ≤ 1.0)
- Careful initialization (Xavier for linear layers)
- Lower learning rate if needed (1e-4 instead of 1e-3)
- Add warmup phase (5k steps)

---

### Risk 3: Insufficient Compute

**Risk:** 100k training steps × 6 mechanisms × 5 settings = 300+ GPU-hours.

**Mitigation:**
- Prioritize: run Experiment 1 (baseline comparison) first
- Use smaller d (10 instead of 20) for initial tests
- Reduce total_steps to 50k for non-critical experiments
- Parallelize: run multiple experiments simultaneously if multi-GPU available

---

### Risk 4: Negative Results

**Risk:** Linear/low-rank attention fail completely; no GD-like ICL observed.

**Mitigation:**
- Negative results are still valuable! Demonstrates necessity of baseline mechanism.
- Analyze failure modes: where exactly does performance degrade?
- Ablation studies: can we fix the mechanism (e.g., add learnable temperature)?
- Contribution to literature: "efficiency approximations harm ICL" is a clear finding

---

### Risk 5: Integration Challenges

**Risk:** Our results don't align with other team members' (sparse, gated, etc.).

**Mitigation:**
- Shared experimental contract enforced from Day 1
- Use same random seeds for evaluation prompts
- Cross-validate: train baseline mechanism ourselves and compare to others' baseline
- Regular sync meetings (2x per week) to catch discrepancies early

---

## Appendix: Mathematical Derivations

### Appendix A: Why Linear Self-Attention ≈ Gradient Descent (Background)

**NOTE:** This appendix explains prior theoretical work on Linear Self-Attention (LSA). LSA is **NOT** one of the mechanisms we're implementing in Q2. This is provided for background context to understand why researchers believe transformers can perform gradient descent.

(From von Oswald et al., 2023)

**Setup:**
- Linear regression: y = Xw*, minimize L(w) = ||y - Xw||²
- Gradient: ∇L(w) = -2X^T(y - Xw)
- GD update: w ← w + η ∇L(w)

**Linear Self-Attention (LSA):**

```
Attention(Q, K, V) = QK^T V

For linear regression prompt:
Q = W_Q X_query
K = W_K X
V = W_V y

Output ∝ X_query^T (X^T X)^{-1} X^T y = X_query^T w_LS
```

**Under specific weight initialization:**
- W_Q W_K^T ∝ I (identity-like)
- W_V projects y appropriately

LSA computes: ŷ = X_query^T (X^T X)^{-1} X^T y

This is equivalent to:
1. Solving w_LS = (X^T X)^{-1} X^T y  (least squares)
2. Predicting ŷ = X_query^T w_LS

Least squares is the solution after infinite GD steps (or one step with optimal step size).

**Key Insight:** LSA implicitly solves a linear system that GD also solves, explaining the alignment.

---

### Appendix B: Low-Rank Attention as Information Bottleneck

**Information Theory View:**

Attention matrix A ∈ R^(n×n) has rank ≤ min(n, d_k).

Constraining rank to r << d_k forces the model to:
1. **Compress** query information into r-dimensional space
2. **Route** information through r "channels"
3. **Reconstruct** output from compressed representation

**Analogy to Autoencoders:**
- Encoder: Q → Q_r (project to rank-r)
- Decoder: Q_r K_r^T → A (reconstruct attention)
- Bottleneck: r dimensions

**Question:** Does the r-dimensional bottleneck preserve enough information to represent gradient descent updates?

**Answer depends on:**
- Task complexity (linear regression is simple)
- Optimal rank of w_LS (often low-rank for structured data)
- Whether attention patterns are inherently low-rank

Our experiments (Exp 5: rank sweep) will empirically answer this.

---

## References

**Core Papers:**

1. Garg et al. (2023). "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes." NeurIPS.

2. von Oswald et al. (2023). "Transformers learn in-context by gradient descent." ICML.

3. Katharopoulos et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML.
   - Introduced kernel-based linear attention with O(n) complexity
   - Showed transformers can be written as RNNs with constant-time inference
   - Used φ(x) = elu(x) + 1 feature map, achieving performance on par with softmax

4. Wang et al. (2020). "Linformer: Self-Attention with Linear Complexity." arXiv.
   - Demonstrated self-attention matrices are empirically low-rank
   - Proved rank k = O(d_k/ε² log n) sufficient for ε-approximation (Theorem 2)
   - Achieved O(n·k) complexity through sequence-length projection E, F

5. Xiong et al. (2021). "Nyströmformer: A Nyström-based Algorithm for Approximating Self-Attention." AAAI.
   - Nyström approximation for low-rank attention

**Background:**

6. Vaswani et al. (2017). "Attention is All You Need." NeurIPS.
   - Original transformer paper

7. Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits." Anthropic.
   - Theoretical analysis of attention mechanisms

8. Olsson et al. (2022). "In-context Learning and Induction Heads." Anthropic.
   - Mechanistic interpretability of ICL

---

## Glossary

**IMPORTANT TERMINOLOGY NOTE:**
- **Linear Self-Attention (LSA)** = Algebraic/mathematical sense, no softmax, QK^T V. **Theoretically proven** equivalent to gradient descent (von Oswald et al.). Not implemented in this project (already studied in Q1).
- **Softmax Attention** = Standard transformer attention with softmax normalization. **Our baseline/control condition.** Shows gradient descent-like behavior empirically (Garg et al., our Q1).
- **Linear Attention (kernel-based)** = Efficiency approximation using kernels, O(n) complexity. **One of our test conditions.** Not the same as LSA above.
- **Low-Rank Attention** = Efficiency approximation with rank constraint. **Our other test condition.**

---

**In-Context Learning (ICL):** Ability of a model to adapt to a task at inference time using only examples in the prompt, without parameter updates.

**Gradient Descent (GD):** Optimization algorithm: w ← w - η ∇L(w).

**GD-1:** One step of gradient descent from initialization w₀ = 0.

**Linear Self-Attention (LSA):** Attention mechanism where Q, K, V are linear functions and softmax is removed, making attention a bilinear operation. Mathematically equivalent to gradient descent under certain conditions (von Oswald et al., 2023). This is NOT what we're implementing—this was studied in Q1.

**Linear Attention (kernel-based):** Attention mechanism using kernel feature maps φ(·) to achieve O(n) complexity in sequence length (Katharopoulos et al., 2020). Replaces softmax with φ(q)^T φ(k) and exploits associativity to avoid computing the n×n attention matrix. **Not the same as LSA** - this uses kernels for efficiency, not for theoretical equivalence to gradient descent.

**Low-Rank Attention:** Attention mechanism where attention matrix A or its components Q, K are constrained to rank r << d.

**Prompt:** Input sequence containing in-context examples and a query: (x₁, y₁, ..., xₙ, yₙ, x_query).

**Cosine Similarity:** cos(u, v) = u·v / (||u|| ||v||), measures alignment between vectors.

**MSE:** Mean squared error: (1/n) Σ(ŷᵢ - yᵢ)².

**Curriculum Learning:** Training strategy where task difficulty gradually increases (e.g., dimension 5 → 20).

**Feature Map:** Function φ: R^d → R^{d'} applied to Q, K in linear attention to enable kernel trick.

---

## Document Metadata

**Version:** 1.0  
**Last Updated:** January 21, 2026  
**Authors:** Krish Prasad  
**Status:** Active - Implementation Phase  
**Review Cycle:** Update after each experimental milestone

**Changelog:**
- v1.0 (2026-01-21): Initial design document covering system architecture, experimental design, and implementation plan
