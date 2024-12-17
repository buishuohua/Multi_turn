# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/16 09:27
    @ Description:
"""

from .BERT_tokenizer import Tokenizer
from dataclasses import dataclass
import torch.nn as nn
import torch

class GloVes:
    # Seems not a trend in Huggingface
    # May change another model

    # dim_50 = Tokenizer(
    #     name="glove_50d",
    #     tokenizer=None,  # Your tokenizer implementation
    #     model=nn.Embedding.from_pretrained(torch.load('path/to/glove_50d'))
    # )
    # dim_100 = Tokenizer(
    #     name="glove_100d",
    #     tokenizer=None,
    #     model=nn.Embedding.from_pretrained(torch.load('path/to/glove_100d'))
    # )
    #
    # @classmethod
    # def get_tokenizer(cls, name: str) -> Tokenizer:
    #     return getattr(cls, name)
    pass
