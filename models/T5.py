# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/16 10:18
    @ Description:
"""
from .BERT_tokenizer import Tokenizer
from dataclasses import dataclass
from transformers import T5Tokenizer, T5Model


class T5s:
    T5_small = Tokenizer(
        name="T5_small",
        tokenizer=T5Tokenizer.from_pretrained("t5-small"),
        model=T5Model.from_pretrained("t5-small")
    )

    T5_base = Tokenizer(
        name="T5_base",
        tokenizer=T5Tokenizer.from_pretrained("t5-base"),
        model=T5Model.from_pretrained("t5-base")
    )

    T5_large = Tokenizer(
        name="T5_large",
        tokenizer=T5Tokenizer.from_pretrained("t5-large"),
        model=T5Model.from_pretrained("t5-large")
    )

    @classmethod
    def get_tokenizer(cls, name: str) -> Tokenizer:
        return getattr(cls, name)
