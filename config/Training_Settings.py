# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 12:59
    @ Description:
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict
import torch
import torch.nn as nn
import os
import re


@dataclass
class TrainingSettings:
    """Specific training configurations"""
    task_type: Literal['Identification', 'Multi', 'Multi_attack'] = 'Multi'
    num_epochs: int = 100
    batch_size: int = 256

    # Optimization settings
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: Optional[float] = 1.0
    optimizer_type: Literal['adam', 'adamw', 'sgd'] = 'adamw'

    # Early stopping and scheduler settings
    early_stopping_patience: int = 5
    scheduler_patience: int = 3
    scheduler_factor: float = 0.1
    min_learning_rate: float = 1e-6

    # Training continuation settings
    continue_training: Optional[bool] = None  # Auto-detect if not specified

    # Checkpoint and saving settings
    checkpoint_freq: int = 10
    save_model_dir: str = 'saved_models'
    save_results_dir: str = 'results'

    # Hardware settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory for fine-tuned embedding models
    fine_tuned_models_dir: str = "fine_tuned_models"

    # Scheduler settings
    scheduler_type: Literal['multi_group', 'reduce_lr_plateau',
                            'cosine', 'linear', 'none'] = 'multi_group'

    # Multi-group scheduler settings
    scheduler_groups: Dict[str, Dict] = field(default_factory=lambda: {
        'embedding': {
            'initial_lr': 5e-5,
            'warmup_steps': 1000,
            'decay_factor': 0.95,
            'min_lr': 1e-6
        },
        'attention': {
            'initial_lr': 1e-4,
            'warmup_steps': 500,
            'decay_factor': 0.9,
            'min_lr': 1e-6
        },
        'other': {
            'initial_lr': 1e-3,
            'warmup_steps': 200,
            'decay_factor': 0.85,
            'min_lr': 1e-7
        }
    })

    class MultiGroupLRScheduler:
        """Custom scheduler for handling different parameter groups with warmup"""

        def __init__(self, optimizer, warmup_steps, decay_factors, min_lrs):
            self.optimizer = optimizer
            self.warmup_steps = warmup_steps  # List of warmup steps for each group
            self.decay_factors = decay_factors  # List of decay factors for each group
            self.min_lrs = min_lrs  # List of minimum learning rates for each group
            self.step_count = 0
            self.best_val_loss = float('inf')
            self.plateau_count = 0

            # Validate inputs
            if len(self.decay_factors) != len(self.optimizer.param_groups):
                raise ValueError(
                    "Number of decay factors must match number of parameter groups")
            if len(self.min_lrs) != len(self.optimizer.param_groups):
                raise ValueError(
                    "Number of minimum learning rates must match number of parameter groups")

            # Store initial learning rates
            self.initial_lrs = [group['lr']
                                for group in optimizer.param_groups]

        def step(self, val_loss=None):
            """
            Update learning rates for all parameter groups

            Args:
                val_loss (float, optional): Current validation loss for plateau detection
            """
            self.step_count += 1

            # Check for plateau if validation loss is provided
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.plateau_count = 0
                else:
                    self.plateau_count += 1

            # Update learning rate for each parameter group
            for group_idx, group in enumerate(self.optimizer.param_groups):
                warmup_steps = self.warmup_steps[group_idx]
                decay_factor = self.decay_factors[group_idx]
                min_lr = self.min_lrs[group_idx]
                initial_lr = self.initial_lrs[group_idx]

                if self.step_count < warmup_steps:
                    # Linear warmup
                    lr = initial_lr * (self.step_count / warmup_steps)
                else:
                    # Exponential decay after warmup
                    steps_after_warmup = self.step_count - warmup_steps
                    lr = initial_lr * (decay_factor ** steps_after_warmup)

                    # Apply plateau reduction if detected
                    if self.plateau_count > 0:
                        # Reduce by 5% for each plateau step
                        lr *= (0.95 ** self.plateau_count)

                # Ensure learning rate doesn't go below minimum
                lr = max(lr, min_lr)

                # Update the learning rate
                group['lr'] = lr

        def get_last_lr(self):
            """Return last computed learning rate for each parameter group"""
            return [group['lr'] for group in self.optimizer.param_groups]

        def state_dict(self):
            """Returns the state of the scheduler as a :class:`dict`"""
            return {
                'current_step': self.current_step,
                'initial_lrs': self.initial_lrs,
                'warmup_steps': self.warmup_steps,
                'decay_factors': self.decay_factors,
                'min_lrs': self.min_lrs
            }

        def load_state_dict(self, state_dict):
            """Loads the scheduler state"""
            self.current_step = state_dict['current_step']
            self.initial_lrs = state_dict['initial_lrs']
            self.warmup_steps = state_dict['warmup_steps']
            self.decay_factors = state_dict['decay_factors']
            self.min_lrs = state_dict['min_lrs']

    def __post_init__(self):
        """Validate settings after initialization and auto-detect continue_training"""
        if self.task_type not in ['Identification', 'Multi', 'Multi_attack']:
            raise ValueError(
                "task_type must be either 'Identification', 'Multi', 'Multi_attack'")

        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")

        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

        if self.continue_training is None:
            self.continue_training = self._auto_detect_continue_training()

        # Validate scheduler settings
        self._validate_scheduler_settings()

    def _auto_detect_continue_training(self) -> bool:
        """Auto-detect whether to continue training based on existing checkpoints"""
        checkpoint_dir = os.path.join(self.save_model_dir)
        if not os.path.exists(checkpoint_dir):
            return False

        # Look for checkpoint files in all experiment subdirectories
        latest_epoch = 0
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pt') and 'epoch_' in file:
                    try:
                        epoch_num = int(
                            re.search(r'epoch_(\d+)', file).group(1))
                        latest_epoch = max(latest_epoch, epoch_num)
                    except (AttributeError, ValueError):
                        continue

        # If we found checkpoints and haven't completed all epochs, continue training
        should_continue = latest_epoch > 0 and latest_epoch < self.num_epochs
        return should_continue

    def get_optimizer(self, model_parameters) -> torch.optim.Optimizer:
        """Return the appropriate optimizer"""

        optimizers = {
            'adam': lambda: torch.optim.Adam(
                model_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            ),
            'adamw': lambda: torch.optim.AdamW(
                model_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            ),
            'sgd': lambda: torch.optim.SGD(
                model_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        }
        return optimizers[self.optimizer_type]()

    def get_scheduler(self, optimizer):
        """Return the appropriate scheduler based on configuration"""
        if self.scheduler_type == 'none':
            return None

        elif self.scheduler_type == 'multi_group':
            # Extract parameters for each group
            warmup_steps = [
                self.scheduler_groups[group]['warmup_steps']
                for group in ['embedding', 'attention', 'other']
            ]
            decay_factors = [
                self.scheduler_groups[group]['decay_factor']
                for group in ['embedding', 'attention', 'other']
            ]
            min_lrs = [
                self.scheduler_groups[group]['min_lr']
                for group in ['embedding', 'attention', 'other']
            ]

            return self.MultiGroupLRScheduler(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                decay_factors=decay_factors,
                min_lrs=min_lrs
            )

        elif self.scheduler_type == 'reduce_lr_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.scheduler_patience,
                factor=self.scheduler_factor,
                min_lr=self.min_learning_rate
            )

        elif self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.min_learning_rate
            )

        elif self.scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.scheduler_factor,
                total_iters=self.num_epochs
            )

    def _validate_scheduler_settings(self):
        """Validate scheduler-related settings"""
        if self.scheduler_type == 'multi_group':
            required_groups = {'embedding', 'attention', 'other'}
            if set(self.scheduler_groups.keys()) != required_groups:
                raise ValueError(
                    f"scheduler_groups must contain exactly: {required_groups}")

            for group, settings in self.scheduler_groups.items():
                required_keys = {'initial_lr',
                                 'warmup_steps', 'decay_factor', 'min_lr'}
                if set(settings.keys()) != required_keys:
                    raise ValueError(
                        f"Group '{group}' must contain exactly: {required_keys}")

                if settings['initial_lr'] <= 0:
                    raise ValueError(
                        f"initial_lr for {group} must be positive")
                if settings['warmup_steps'] < 0:
                    raise ValueError(
                        f"warmup_steps for {group} must be non-negative")
                if not 0 < settings['decay_factor'] <= 1:
                    raise ValueError(
                        f"decay_factor for {group} must be between 0 and 1")
                if settings['min_lr'] <= 0:
                    raise ValueError(f"min_lr for {group} must be positive")
                if settings['min_lr'] >= settings['initial_lr']:
                    raise ValueError(
                        f"min_lr must be less than initial_lr for {group}")

    @classmethod
    def get_default(cls):
        """Return default training settings"""
        return cls(
            num_epochs=100,
            batch_size=256,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=1.0,
            optimizer_type='adamw',
            continue_training=True,
            scheduler_type='multi_group',
            scheduler_groups={
                'embedding': {
                    'initial_lr': 5e-5,
                    'warmup_steps': 1000,
                    'decay_factor': 0.95,
                    'min_lr': 1e-6
                },
                'attention': {
                    'initial_lr': 1e-4,
                    'warmup_steps': 500,
                    'decay_factor': 0.9,
                    'min_lr': 1e-6
                },
                'other': {
                    'initial_lr': 1e-3,
                    'warmup_steps': 200,
                    'decay_factor': 0.85,
                    'min_lr': 1e-7
                }
            }
        )
