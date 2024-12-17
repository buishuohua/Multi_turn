# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 13:25
    @ Description:
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class ModelSettings:
    """Model architecture settings"""
    # Required parameters
    input_dim: int
    output_dim: int
    embedding_dim: int
    embedding_type: Literal['BERT_base_uncased', 'BERT_large_uncased',
    'RoBERTa_base', 'glove_100d', 'glove_300d']
    lags: int  # Number of time steps to consider

    # Optional parameters with defaults
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.1
    loss: Literal['categorical_cross_entropy', 'binary_cross_entropy',
    'mse', 'mae'] = 'categorical_cross_entropy'
    activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = 'relu'
    output_activation: Literal['softmax', 'sigmoid', 'linear'] = 'softmax'

    def __post_init__(self):
        """Validate settings after initialization"""
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        if not all(isinstance(dim, int) and dim > 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive integers")

        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

