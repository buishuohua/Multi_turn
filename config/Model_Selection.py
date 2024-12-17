# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 12:44
    @ Description:
"""

from dataclasses import dataclass
from typing import List, Optional, Literal


@dataclass
class ModelSelection:
    """Settings for model type selection"""
    # Need to be fixed, suppose 2 or 3 total types and then multiple subtypes
    model_type: Literal['BiLSTM', 'CNNBiLSTM', 'AttentionBiLSTM'] = 'BiLSTM'
    use_attention: bool = False
    use_cnn: bool = False
    cnn_layers: int = 2
    cnn_kernel_size: int = 3
    attention_heads: int = 8