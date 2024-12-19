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
from transformers import AutoConfig

@dataclass
class ModelSettings:
    """Model architecture settings"""
    output_dim: Optional[int] = None  # Make it optional, will be set by Trainer
    embedding_dim: Optional[int] = None  # Now optional
    embedding_type: Literal['BERT_base_uncased', 'BERT_large_uncased',
                            'RoBERTa_base', 'glove_100d', 'glove_300d'] = 'BERT_base_uncased'

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
                         'orthogonal', 'normal', 'uniform'] = 'orthogonal'
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
        # First validate and set embedding dimension
        self._validate_and_set_embedding_dim()

        # Then validate other settings
        self._validate_settings()

        # Original post_init logic
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        self.hidden_dims = [self.init_hidden_dim //
                            (2**i) for i in range(self.num_layers)]

        if not all(isinstance(dim, int) and dim > 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive integers")

        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")

    def _get_model_name(self) -> str:
        """Convert embedding type to huggingface model name"""
        model_names = {
            'BERT_base_uncased': 'bert-base-uncased',
            'BERT_large_uncased': 'bert-large-uncased',
            'RoBERTa_base': 'roberta-base',
            'RoBERTa_large': 'roberta-large',
        }
        if self.embedding_type not in model_names:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        return model_names[self.embedding_type]

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension based on model type"""
        try:
            # Handle transformer models
            if any(model_type in self.embedding_type
                   for model_type in ['BERT', 'RoBERTa']):
                config = AutoConfig.from_pretrained(self._get_model_name())
                return config.hidden_size

            # Handle GloVe embeddings
            elif 'glove' in self.embedding_type.lower():
                try:
                    return int(self.embedding_type.lower().split('_')[1].replace('d', ''))
                except (IndexError, ValueError):
                    raise ValueError(
                        f"Invalid GloVe embedding format. Expected format: 'glove_<dim>d'"
                    )
            else:
                raise ValueError(
                    f"Unknown embedding type: {self.embedding_type}")

        except Exception as e:
            raise ValueError(f"Error getting embedding dimension: {str(e)}")

    def _validate_and_set_embedding_dim(self):
        """Validate and set embedding dimension"""
        try:
            # Get the model's actual embedding dimension
            model_embedding_dim = self._get_embedding_dim()

            # If embedding_dim is not provided, use the model's dimension
            if self.embedding_dim is None:
                self.embedding_dim = model_embedding_dim
            # If embedding_dim is provided, validate it matches the model
            elif self.embedding_dim != model_embedding_dim:
                raise ValueError(
                    f"Provided embedding_dim ({self.embedding_dim}) does not match "
                    f"{self.embedding_type}'s dimension ({model_embedding_dim})"
                )
        except Exception as e:
            raise ValueError(f"Error validating embedding dimension: {str(e)}")

    def _validate_settings(self):
        """Validate all model settings"""
        errors = []

        # Validate dropout rate
        if not 0 <= self.dropout_rate <= 1:
            errors.append("dropout_rate must be between 0 and 1")

        # Validate dimensions
        if self.output_dim <= 0:
            errors.append("output_dim must be positive")

        if self.init_hidden_dim <= 0:
            errors.append("init_hidden_dim must be positive")

        # Validate number of layers
        if self.num_layers <= 0:
            errors.append("num_layers must be positive")

        # Validate initialization parameters
        if self.init_std <= 0:
            errors.append("init_std must be positive")

        if self.init_b <= self.init_a:
            errors.append("init_b must be greater than init_a")

        # If there are any errors, raise them all at once
        if errors:
            raise ValueError("\n".join(errors))

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

    @classmethod
    def get_default(cls):
        return cls(
            output_dim=None,  # Will be set automatically by Trainer
            embedding_dim=None,  # Will be set based on embedding type
            embedding_type='BERT_base_uncased',
            init_hidden_dim=256,
            dropout_rate=0.1,
            num_layers=2,
            bidirectional=True,
            fine_tune_embedding=False,
            activation='relu',
            final_activation='softmax',
            loss='cross_entropy',
            pooling='last',
            weight_init='orthogonal',
            init_gain=1.0,
            init_mean=0.0,
            init_std=0.02,
            init_a=-0.1,
            init_b=0.1,
            focal_alpha=1.0,
            focal_gamma=2.0,
            label_smoothing=0.1,
            class_weights=None
        )
