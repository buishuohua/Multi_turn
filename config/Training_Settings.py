# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 12:59
    @ Description:
"""

from dataclasses import dataclass
from typing import Optional, Literal
import torch
import torch.nn as nn
import os
import re


@dataclass
class TrainingSettings:
    """Specific training configurations"""
    num_epochs: int = 100
    batch_size: int = 256

    # Optimization settings
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: Optional[float] = 1.0
    optimizer_type: Literal['adam', 'adamw', 'sgd'] = 'adam'

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

    def __post_init__(self):
        """Validate settings after initialization and auto-detect continue_training"""
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
        if should_continue:
            print(f"Found existing checkpoint at epoch {latest_epoch}/{self.num_epochs}. "
                  f"Training will continue from the latest checkpoint.")
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
        """Return the learning rate scheduler"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            min_lr=self.min_learning_rate
        )

    @classmethod
    def get_default(cls):
        return cls(
            num_epochs=100,
            batch_size=256,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=1.0,
            optimizer_type='adam',
        )
