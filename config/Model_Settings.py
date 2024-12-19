# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 13:25
    @ Description:
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union, Dict
import torch.nn as nn
import torch


@dataclass
class ModelSettings:
    """Model architecture settings"""
    output_dim: int
    embedding_dim: int
    embedding_type: Literal['BERT_base_uncased', 'BERT_large_uncased',
                            'RoBERTa_base', 'glove_100d', 'glove_300d']

    # Optional parameters (with defaults)
    init_hidden_dim: int = 256
    hidden_dims: List[int] = field(init=False)
    dropout_rate: float = 0.1
    num_layers: int = 2
    bidirectional: bool = True
    fine_tune_embedding: bool = False
    activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = 'relu'
    final_activation: Literal['softmax', 'sigmoid', 'linear'] = 'softmax'
    loss: Literal['cross_entropy',              # Standard cross-entropy
                  # BCE with logits (recommended for multi-label)
                  'bce_with_logits',
                  'weighted_cross_entropy',      # For class imbalance
                  'focal',                       # For highly imbalanced datasets
                  'label_smoothing_ce',         # Cross-entropy with label smoothing
                  'kl_div',                     # KL divergence loss
                  'mse',                        # Mean squared error
                  'mae'] = 'cross_entropy'      # Mean absolute error
    pooling: Literal['last', 'mean'] = 'last'
    weight_init: Literal['xavier_uniform', 'xavier_normal',
                         'kaiming_uniform', 'kaiming_normal',
                         'orthogonal', 'normal', 'uniform'] = 'xavier_uniform'
    init_gain: float = 1.0  # Gain parameter for xavier/orthogonal init
    init_mean: float = 0.0  # For normal initialization
    init_std: float = 0.02  # For normal initialization
    init_a: float = -0.1   # For uniform initialization lower bound
    init_b: float = 0.1    # For uniform initialization upper bound

    # Add new loss-related parameters
    focal_alpha: float = 1.0                    # Focal loss alpha parameter
    focal_gamma: float = 2.0                    # Focal loss gamma parameter
    label_smoothing: float = 0.1               # Label smoothing factor
    # Class weights for weighted losses
    class_weights: Optional[List[float]] = None

    def __post_init__(self):
        """Validate settings after initialization"""
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        self.hidden_dims = [self.init_hidden_dim //
                            (2**i) for i in range(self.num_layers)]

        if not all(isinstance(dim, int) and dim > 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive integers")

        if self.embedding_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

    def get_activation(self) -> nn.Module:
        """Return the appropriate activation function"""

        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations[self.activation]

    def get_final_activation(self) -> Optional[nn.Module]:
        """Return the final activation function"""
        if self.final_activation == 'none':
            return None
        activations = {
            'softmax': nn.Softmax(dim=1),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(self.final_activation)

    def get_loss(self) -> nn.Module:
        """Return the appropriate loss function"""
        if self.loss == 'cross_entropy':
            return nn.CrossEntropyLoss(
                weight=torch.tensor(
                    self.class_weights) if self.class_weights else None
            )

        elif self.loss == 'bce_with_logits':
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    self.class_weights) if self.class_weights else None
            )

        elif self.loss == 'weighted_cross_entropy':
            if not self.class_weights:
                raise ValueError(
                    "class_weights must be provided for weighted_cross_entropy")
            return nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights))

        elif self.loss == 'focal':
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1, gamma=2):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma

                def forward(self, inputs, targets):
                    ce_loss = nn.CrossEntropyLoss()(inputs, targets)
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss

            return FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)

        elif self.loss == 'label_smoothing_ce':
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        elif self.loss == 'kl_div':
            return nn.KLDivLoss(reduction='batchmean')

        elif self.loss == 'mse':
            return nn.MSELoss()

        elif self.loss == 'mae':
            return nn.L1Loss()

        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def get_initializer(self) -> Dict:
        """Return initialization configuration"""
        init_config = {
            'method': self.weight_init,
            'params': {
                'gain': self.init_gain,
                'mean': self.init_mean,
                'std': self.init_std,
                'a': self.init_a,
                'b': self.init_b
            }
        }
        return init_config
