"""
Training script for in-context learning models.
"""
import os
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.samplers import GaussianSampler
from src.data.tasks import LinearRegressionTask, mean_squared_error
from src.data.curriculum import Curriculum
from src.models.transformer import InContextTransformer


class Trainer:
    """Trainer class for in-context learning models."""
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary with training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = InContextTransformer(
            n_dims=config['n_dims'],
            n_positions=config['max_n_points'],
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head']
        ).to(self.device)
        
        print(f"Model has {self.model.count_parameters():,} parameters")
        print(f"Using device: {self.device}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Initialize data sampler
        self.data_sampler = GaussianSampler(n_dims=config['n_dims'])
        
        # Initialize curriculum
        if config['use_curriculum']:
            self.curriculum = Curriculum(
                dims_start=config['curriculum']['dims_start'],
                dims_end=config['curriculum']['dims_end'],
                dims_inc=config['curriculum']['dims_inc'],
                dims_interval=config['curriculum']['dims_interval'],
                points_start=config['curriculum']['points_start'],
                points_end=config['curriculum']['points_end'],
                points_inc=config['curriculum']['points_inc'],
                points_interval=config['curriculum']['points_interval']
            )
        else:
            # No curriculum - use full complexity from start
            self.curriculum = Curriculum(
                dims_start=config['n_dims'],
                dims_end=config['n_dims'],
                dims_inc=0,
                dims_interval=1,
                points_start=config['max_n_points'],
                points_end=config['max_n_points'],
                points_inc=0,
                points_interval=1
            )
        
        self.start_step = 0
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint if exists
        if config.get('resume', True):
            self.load_checkpoint()
    
    def train_step(self, xs, ys):
        """
        Single training step.
        
        Args:
            xs: Input tensor (batch_size, n_points, n_dims)
            ys: Output tensor (batch_size, n_points)
        
        Returns:
            loss: Training loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass - use true ys for training (model predicts at y positions)
        # The model architecture predicts at y positions (1, 3, 5, ...)
        # where it has seen the corresponding x but not the y yet (causal masking)
        predictions = self.model(xs, ys)
        
        # Compute loss - this is correct, we compare predictions to true ys
        loss = mean_squared_error(predictions, ys)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop."""
        pbar = tqdm(
            range(self.start_step, self.config['num_steps']),
            desc='Training',
            initial=self.start_step,
            total=self.config['num_steps']
        )
        
        log_interval = self.config.get('log_interval', 100)
        save_interval = self.config.get('save_interval', 10000)
        
        for step in pbar:
            # Sample data
            n_points = self.curriculum.n_points
            n_dims_truncated = self.curriculum.n_dims_truncated
            batch_size = self.config['batch_size']
            
            # Sample inputs
            xs = self.data_sampler.sample_xs(
                n_points=n_points,
                batch_size=batch_size,
                n_dims_truncated=n_dims_truncated,
                device=self.device
            )
            
            # Sample task and evaluate
            task = LinearRegressionTask(
                n_dims=self.config['n_dims'],
                batch_size=batch_size,
                n_dims_truncated=n_dims_truncated,
                device=self.device
            )
            ys = task.evaluate(xs)
            
            # Training step
            loss = self.train_step(xs, ys)
            
            # Update curriculum
            self.curriculum.update()
            
            # Logging
            if step % log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss:.6f}',
                    'dims': self.curriculum.n_dims_truncated,
                    'points': self.curriculum.n_points
                })
                
                # Log to file
                log_entry = {
                    'step': step,
                    'loss': loss,
                    'n_dims_truncated': self.curriculum.n_dims_truncated,
                    'n_points': self.curriculum.n_points
                }
                self.log(log_entry)
            
            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                self.save_checkpoint(step)
        
        # Final checkpoint
        self.save_checkpoint(self.config['num_steps'])
        print(f"Training complete! Final checkpoint saved.")
    
    def save_checkpoint(self, step):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        step_checkpoint_path = self.checkpoint_dir / f'checkpoint_{step}.pt'
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'curriculum_state': self.curriculum.get_state(),
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save periodic checkpoints
        if step % (self.config.get('save_interval', 10000) * 5) == 0:
            torch.save(checkpoint, step_checkpoint_path)
        
        print(f"\nCheckpoint saved at step {step}")
    
    def load_checkpoint(self):
        """Load training checkpoint if exists."""
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_step = checkpoint['step']
            
            # Restore curriculum state
            curriculum_state = checkpoint['curriculum_state']
            self.curriculum.n_dims_truncated = curriculum_state['n_dims_truncated']
            self.curriculum.n_points = curriculum_state['n_points']
            self.curriculum.step_count = curriculum_state['step_count']
            
            print(f"Resumed from step {self.start_step}")
        else:
            print("No checkpoint found, starting from scratch")
    
    def log(self, log_entry):
        """Append log entry to log file."""
        log_file = self.checkpoint_dir / 'training_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train in-context learning model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_steps', type=int, help='Number of training steps (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override num_steps if provided
    if args.num_steps is not None:
        config['num_steps'] = args.num_steps
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
