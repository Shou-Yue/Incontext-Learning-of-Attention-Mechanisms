from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # Attention mechanism
    attn_type: str = 'baseline'  # {'baseline', 'linear', 'lowrank'}
    attn_rank: int = 16          # Only used for lowrank
    attn_feature_map: str = 'elu'  # Only used for linear
    
    # Model architecture
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_mlp: int = 1024
    dropout: float = 0.0
    
    # Task parameters
    task_dim: int = 20
    max_prompt_length: int = 40
    noise_std: float = 0.0
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    total_steps: int = 100_000
    warmup_steps: int = 5_000
    
    # Curriculum
    curriculum_start_dim: int = 5
    curriculum_end_step: int = 20_000
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    
    # Paths
    checkpoint_dir: str = 'results/checkpoints'
    log_dir: str = 'results/logs'
    
    # Reproducibility
    seed: int = 42
