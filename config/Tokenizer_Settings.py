# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 12:49
    @ Description:
"""

from dataclasses import dataclass
from typing import Literal, Optional
from models.Embedding.BERT_tokenizer import BERTs
from models.Embedding.GloVe import GloVes


@dataclass
class TokenizerSettings:
    name: str  # e.g., "BERT_base_uncased" or "glove_100d"
    truncation: str
    embedding_type: Literal['bert', 'glove', 'word2vec']
    max_length: int = 512
    padding: bool = True
    add_special_tokens: bool = True

    def get_model(self):
        if self.embedding_type == 'bert':
            return BERTs.get_tokenizer(self.name)
        elif self.embedding_type == 'glove':
            return GloVes.get_tokenizer(self.name)
