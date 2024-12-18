# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/14 12:24
    @ Description:
"""

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from typing import Any


@dataclass
class Tokenizer:
    name: str
    tokenizer: Any
    model: Any


class BERTs:
    BERT_base_uncased = Tokenizer(
        name="BERT_base_uncased",
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        model=AutoModel.from_pretrained("bert-base-uncased")
    )
    BERT_large_uncased = Tokenizer(
        name="BERT_large_uncased",
        tokenizer=AutoTokenizer.from_pretrained("bert-large-uncased"),
        model=AutoModel.from_pretrained("bert-large-uncased")
    )

    @classmethod
    def get_tokenizer(cls, name: str) -> Tokenizer:
        return getattr(cls, name)
