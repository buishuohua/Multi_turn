# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 12:44
    @ Description:
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Literal, Type
import torch.nn as nn

if TYPE_CHECKING:
    from models.LSTM.BiLSTM import BiLSTM
    from models.LSTM.CNNBiLSTM import CNNBiLSTM
else:
    BiLSTM = CNNBiLSTM = None

@dataclass
class ModelSelection:
    """Settings for model type selection"""
    model_type: Literal['BiLSTM', 'CNNBiLSTM'] = 'BiLSTM'
    
    def get_model(self, config) -> nn.Module:
        """Create and return the appropriate model instance"""
        # Import here to avoid circular imports
        from models.LSTM.BiLSTM import BiLSTM
        from models.LSTM.CNNBiLSTM import CNNBiLSTM
        
        MODEL_REGISTRY = {
            'BiLSTM': BiLSTM,
            'CNNBiLSTM': CNNBiLSTM,
        }
        
        if self.model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model_class = MODEL_REGISTRY[self.model_type]
        return model_class(config)

    @classmethod
    def get_default(cls):
        return cls(model_type='BiLSTM')
