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
    max_length: Optional[int] = None
    padding: bool = True
    add_special_tokens: bool = True

    def __post_init__(self):
        MAX_LENGTHS = {
            'BERT_base_uncased': 512,
            'BERT_large_uncased': 512,
            'RoBERTa_base': 512,
            'glove_100d': 1000,
            'glove_300d': 1000
        }

        if self.max_length is None:
            self.max_length = MAX_LENGTHS.get(self.name, 512)

        max_allowed = MAX_LENGTHS.get(self.name)
        if max_allowed and self.max_length > max_allowed:
            raise ValueError(
                f"max_length ({self.max_length}) exceeds maximum allowed length "
                f"({max_allowed}) for tokenizer {self.name}"
            )

    def get_model(self):
        if self.embedding_type == 'bert':
            return BERTs.get_tokenizer(self.name)
        elif self.embedding_type == 'glove':
            return GloVes.get_tokenizer(self.name)
        else:
            raise ValueError(
                f"Unsupported embedding type: {self.embedding_type}")

    @classmethod
    def get_default(cls):
        return cls(
            name='BERT_base_uncased',
            truncation='ratio',
            embedding_type='bert',
            max_length=None,
            padding=True,
            add_special_tokens=True
        )

    def get_tokenizer_info(self) -> dict:
        return {
            'name': self.name,
            'embedding_type': self.embedding_type,
            'max_length': self.max_length,
            'truncation': self.truncation,
            'padding': self.padding,
            'add_special_tokens': self.add_special_tokens
        }
