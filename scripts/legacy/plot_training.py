#!/usr/bin/env python3
"""
Plot training progress from log file.
Usage: python scripts/plot_training.py checkpoints/test_local/training_log.jsonl
"""
import json
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python scripts/plot_training.py <log_file.jsonl>")
    sys.exit(1)

log_file = sys.argv[1]

steps = []
losses = []
dims = []
points = []

with open(log_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        steps.append(data['step'])
        losses.append(data['loss'])
        dims.append(data['n_dims_truncated'])
        points.append(data['n_points'])

# Create plots
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Loss
axes[0].plot(steps, losses)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True)

# Dimensions
axes[1].plot(steps, dims)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('N Dims')
axes[1].set_title('Curriculum: Dimensions')
axes[1].grid(True)

# Points
axes[2].plot(steps, points)
axes[2].set_xlabel('Step')
axes[2].set_ylabel('N Points')
axes[2].set_title('Curriculum: In-Context Examples')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150)
print(f"Saved plot to training_progress.png")
print(f"\nFinal stats:")
print(f"  Steps: {steps[-1]}")
print(f"  Loss: {losses[-1]:.6f}")
print(f"  Dims: {dims[-1]}")
print(f"  Points: {points[-1]}")
