# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/16 12:14
    @ Description:
"""

from dataclasses import dataclass


@dataclass
class DataSettings:

    which: str = "question",
    random_state: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
