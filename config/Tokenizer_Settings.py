# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 12:49
    @ Description:
"""

from dataclasses import dataclass
from typing import Literal, Optional
from models.Embedding.BERT_tokenizer import BERTs
from models.Embedding.T5 import T5s


@dataclass
class TokenizerSettings:
    name: str  # e.g., "BERT_base_uncased" or "glove_100"
    truncation: str
    embedding_type: Literal['bert', 't5']
    max_length: Optional[int] = None
    padding: bool = True
    add_special_tokens: bool = True

    def _get_base_type(self, model_name: str) -> str:
        """Map full model name to base type"""
        if any(name in model_name.lower() for name in ['bert', 'roberta']):
            return 'bert'
        elif 't5' in model_name.lower():
            return 't5'
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def __post_init__(self):
        MAX_LENGTHS = {
            'BERT_base_uncased': 512,
            'BERT_large_uncased': 512,
            'BERT_base_multilingual_cased': 512,
            'XLM_roberta_large': 512,
            'T5_small': 512,
            'T5_base': 512,
            'T5_large': 512,
        }

        if self.max_length is None:
            self.max_length = MAX_LENGTHS.get(self.name, 512)

        max_allowed = MAX_LENGTHS.get(self.name)
        if max_allowed and self.max_length > max_allowed:
            raise ValueError(
                f"max_length ({self.max_length}) exceeds maximum allowed length "
                f"({max_allowed}) for tokenizer {self.name}"
            )

        # Automatically set embedding_type based on name
        self.embedding_type = self._get_base_type(self.name)

    def get_model(self):
        if self.embedding_type == 'bert':
            return BERTs.get_tokenizer(self.name)
        elif self.embedding_type == 't5':
            return T5s.get_tokenizer(self.name)
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
