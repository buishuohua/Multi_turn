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
    BERT_base_multilingual_cased = Tokenizer(
        name="BERT_base_multilingual_cased",
        tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
        model=AutoModel.from_pretrained("bert-base-multilingual-cased")
    )

    XLM_roberta_base = Tokenizer(
        name="XLM_roberta_base",
        tokenizer=AutoTokenizer.from_pretrained("xlm-roberta-base"),
        model=AutoModel.from_pretrained("xlm-roberta-base")
    )
    
    XLM_roberta_large = Tokenizer(
        name="XLM_roberta_large",
        tokenizer=AutoTokenizer.from_pretrained("xlm-roberta-large"),
        model=AutoModel.from_pretrained("xlm-roberta-large")
    )

    @classmethod
    def get_tokenizer(cls, name: str) -> Tokenizer:
        return getattr(cls, name)
