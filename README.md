# Q2 Attention Mechanism Experiments for In-Context Learning

This repository is a clean rebuild focused on the Q2 proposal:

- Compare `softmax` (baseline), `linear` (kernelized), and `lowrank` attention.
- Evaluate whether optimization-like in-context learning is preserved.
- Run controlled linear-regression ICL experiments with shared data/task settings.

## Implemented Scope

- Swappable attention modules:
  - `src/models/attention/baseline_attention.py`
  - `src/models/attention/linear_attention.py`
  - `src/models/attention/lowrank_attention.py`
- Transformer backbone:
  - `src/models/transformer.py`
- Synthetic task generation + prompt formatting:
  - `src/data/task_sampler.py`
- Training loop with curriculum, eval, checkpointing:
  - `src/training/trainer.py`
- Baselines + GD-alignment metrics:
  - `src/evaluation/baselines.py`
  - `src/evaluation/gd_alignment.py`
  - `src/evaluation/metrics.py`
- Experiment scripts:
  - `scripts/exp1_baseline_comparison.py`
  - `scripts/exp2_scaling.py`
  - `scripts/exp3_noise_robustness.py`
  - `scripts/exp4_distribution_shift.py`
  - `scripts/exp5_lowrank_sweep.py`
  - `scripts/exp6_attention_analysis.py`

## Setup

```bash
pip install -r requirements.txt
```

## Train One Model

```bash
python scripts/train.py --config configs/train_baseline_d20.json
python scripts/train.py --config configs/train_linear_d20.json
python scripts/train.py --config configs/train_lowrank_d20_r16.json
```

## Evaluate One Checkpoint

```bash
python scripts/evaluate.py \
  --config configs/train_baseline_d20.json \
  --n_context 40
```

By default, `scripts/evaluate.py` uses `checkpoint_best.pt` in the config output directory.

## Run Proposal Experiments

```bash
python scripts/exp1_baseline_comparison.py
python scripts/exp2_scaling.py
python scripts/exp3_noise_robustness.py
python scripts/exp4_distribution_shift.py
python scripts/exp5_lowrank_sweep.py
python scripts/exp6_attention_analysis.py
```

Or run all in sequence:

```bash
bash scripts/run_all_experiments.sh
```

## Quick Smoke Test

```bash
python scripts/smoke_test.py
```

## Notes on Design Choices

- Prompt format is interleaved: `[x1, y1, x2, y2, ..., xn, yn, x_query]`.
- Training objective predicts `y_query` from the final query token.
- Linear attention uses `phi(x) = elu(x) + 1` and causal prefix-sum implementation.
- Low-rank attention uses strict non-causal Linformer-style sequence projections `E, F` (n -> r).
- Default training loss is sequence-wide over all y slots (Garg-style objective).
- GD alignment uses implicit weight extraction from basis-vector queries.

## Tests

```bash
pytest tests -q
```
