# Grouped Query Attention (GQA) – In-Context Learning Experiments

This branch contains the implementation and experiments for **Grouped Query Attention (GQA)** as part of our team project:

**In-Context Learning of Attention Mechanisms**

This branch focuses specifically on evaluating how **Grouped Query Attention** affects a transformer's ability to perform in-context learning on synthetic regression tasks.

---

## 1. Project Overview and Motivation

The goal of this project is to analyze how different attention mechanisms affect a transformer's ability to perform **in-context learning (ICL)**.

In-context learning refers to a model’s ability to infer a task from examples provided in the input sequence — without updating model weights.

Recent theoretical work (Oswald et al., 2023) suggests that the structure of the attention mechanism plays a key role in enabling gradient descent–like behavior inside transformers.

Our central research question is:

> How does modifying the attention mechanism change a transformer's ability to perform in-context learning?

To answer this, each team member implements and evaluates a transformer using a different attention variant under the same experimental setup. This branch analyzes the behavior of **Grouped Query Attention (GQA)** within that controlled framework.

---

## 2. What is Grouped Query Attention (GQA)?

In standard multi-head attention:

- Each attention head has its own **Query (Q)**, **Key (K)**, and **Value (V)** projections.

In **Grouped Query Attention (GQA)**:

- We maintain multiple **query heads**
- But use fewer **key/value heads**
- Key and value projections are shared across groups of query heads

Formally:

- Let `num_q_heads` be the number of query heads  
- Let `num_kv_heads` be the number of key/value heads  
- We enforce: num_q_heads % num_kv_heads == 0


Each KV head is shared across `num_q_heads / num_kv_heads` query heads.

### Why use GQA?

- Reduces memory and computation compared to full multi-head attention
- Maintains expressive query diversity
- Often retains strong performance with improved efficiency

In this branch, we implement GQA inside a transformer and evaluate its performance on synthetic in-context learning tasks.

---

## 3. Experimental Framework

We follow the synthetic linear regression in-context learning setup from Oswald et al. (2023).

### Task Setup

- Input dimension: `d = 20`
- Number of in-context examples: `n = 2d + 1 = 41`
- Noise level: `σ = 0.1`
- Batch size: `64`

Each task consists of:

- `n` in-context (x, y) pairs
- A new query input `x_q`
- The model must predict the corresponding `y_q`

---

## 4. Metrics

At evaluation checkpoints, we compute:

- **MSE (GQA)**  
Mean squared error of the GQA transformer on new tasks.

- **MSE (GD baseline)**  
Mean squared error of true gradient descent applied to the same tasks.

- **MSE**  
Error using the ground-truth regression weights.

- **Cosine similarity (GQA vs GD)**  
Measures how closely the GQA model’s predictions align with gradient descent predictions.

Cosine similarity interpretation:

- ~0 → unrelated behavior  
- ~0.3–0.5 → partially aligned  
- ~1.0 → strongly GD-like  

---

## 5. Experiment 1 — Training Steps Sweep

### Objective

Test whether GQA learns gradient descent–like behavior with increasing training.

> **x = training steps (0 → 100k)**

### Fixed Architecture

- `num_layers = 8`
- `num_q_heads = 4`
- `num_kv_heads = 1`
- Hidden dimension = 64

### Procedure

The model is trained for 100,000 steps.

At checkpoints (0, 1k, 5k, 10k, 25k, 50k, 75k, 100k), we evaluate:

- MSE (GQA)
- MSE (GD baseline)
- MSE (Oracle)
- Cosine similarity between GQA and GD predictions

### Results Summary

We observe:

- GQA initially underperforms GD.
- With training, GQA surpasses GD in task MSE.
- Cosine similarity increases from ~0 at initialization to ~0.4 by 100k steps.

This indicates that GQA:

- Successfully learns the regression task.
- Partially approximates gradient descent behavior.
- Does not converge to a strict GD emulator (cosine ≪ 1).

---

## 6. Repository Structure


dhanvi/
│
└───src                    
│    │ 
│    └───data                      
│    │      └───linear_regression.py # Synthetic regression task generation
│    │ 
│    └───models                     
│    │      └───gqa.py # GQA transformer implementation
│    │ 
│    └───evaluation                 
│    │      └───gd_baseline.py # Gradient descent baseline + cosine metric
│    │ 
│    │ 
└───scripts              
│    │ 
│    └─── train_gqa_steps.py # Experiment 1: Training steps sweep                  
│    └─── plot_gqa_steps.py # Plotting for Experiment 1
│    │ 
└─── results 
│    └─── gqa_steps/ # Saved metrics + figures
│    
└───README.md
    
---

## 7. Reproducing Experiment 1 - 100k train steps 

### Step 1 — Clone the repository

```bash
git clone https://github.com/Shou-Yue/Incontext-Learning-of-Attention-Mechanisms.git
cd Incontext-Learning-of-Attention-Mechanisms
git checkout dhanvi
```

### Step 2 — Install dependencies 

```bash
pip install torch numpy matplotlib
```

### Step 3 — Run training

```bash
python -m scripts.train_gqa_steps
```

saves metrics ro results/gqa_steps/metrics_steps.npz

### Step 4 - Generate plots 

```bash
python -m scripts.plot_gqa_steps
```

