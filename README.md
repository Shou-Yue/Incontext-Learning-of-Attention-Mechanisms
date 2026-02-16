## Grouped Query Attention (GQA) – In-Context Learning Experiments

This branch contains the implementation and experiments for **Grouped Query Attention (GQA)** as part of our team project:

**In-Context Learning of Attention Mechanisms**

This branch focuses specifically on **Grouped Query Attention**.

---

## 1. Project Overview and Motivation

The goal of this project is to analyze how different attention mechanisms affect a transformer's ability to perform **in-context learning (ICL)** on synthetic regression tasks.

In-context learning refers to a model’s ability to infer a task from examples provided in the input sequence — without updating model weights.

Recent theoretical work suggests that the structure of the attention mechanism plays a key role in enabling in-context learning behavior in transformers.

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
- Let `n_heads` be the number of query heads
- Let `n_kv_heads` be the number of key/value heads
- We enforce:  
  `n_heads % n_kv_heads == 0`
- Each KV head is shared across `n_heads / n_kv_heads` query heads

### Why use GQA?

- Reduces memory and computation compared to full multi-head attention
- Keeps expressive query diversity
- Often retains similar performance with improved efficiency

In this branch, we implement GQA inside a transformer and evaluate its performance on synthetic in-context learning tasks.

---

## 3. Repository Structure (This Branch)

```
src/
  gqa_experiments.ipynb    # End-to-end experiment notebook
```

The notebook contains:
1. GQA attention implementation
2. Transformer architecture
3. Synthetic task generation
4. Training loop
5. Evaluation (MSE)
6. Visualization of learning curves

---

## 4. Reproducibility Instructions

### Step 1: Clone the repository
```bash
git clone https://github.com/Shou-Yue/Incontext-Learning-of-Attention-Mechanisms.git
cd Incontext-Learning-of-Attention-Mechanisms
git checkout dhanvi
```

### Step 2: Install dependencies
If using a shared team environment, activate that.

Otherwise:
```bash
pip install torch numpy pandas matplotlib jupyter
```

### Step 3: Run the notebook
```bash
jupyter notebook src/gqa_experiments.ipynb
```
