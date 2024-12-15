# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 13:25
    @ Description:
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelSettings:

    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    embedding_dim: int
    embedding_type: str
    lags: int
    batch_size: int
    max_tokens: int
    dropout_rate: float = 0.1
    activation: str = "relu"
    batch_size: int = 32
    learning_rate: float = 0.001
    output_activation: str = "softmax"

