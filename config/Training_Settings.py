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


@dataclass
class TrainingSettings:
    """Specific training configurations"""
    num_epochs: int = 100
    batch_size: int = 32

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

    # Checkpoint and saving settings
    checkpoint_freq: int = 10
    mode: str = 'train'
    save_model_dir: str = 'saved_models'
    save_results_dir: str = 'results'
    experiment_name: str = f'{mode}'

    # Hardware settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        """Validate settings after initialization"""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")

        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

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

