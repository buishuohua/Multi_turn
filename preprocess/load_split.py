# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 10:03
    @ Description:
"""

import os
import sys
import ast
import pandas as pd
from sklearn.model_selection import train_test_split


def loader():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")

    actor_file = "gather_actor_attack_results.csv"
    coa_file = "gather_coa_attack_results.csv"
    safe_file = "gather_safe_results.csv"
    actor_path = os.path.join(DATA_PATH, actor_file)
    coa_path = os.path.join(DATA_PATH, coa_file)
    safe_path = os.path.join(DATA_PATH, safe_file)

    actor_origin = pd.read_csv(actor_path)
    coa_origin = pd.read_csv(coa_path)
    safe_origin = pd.read_csv(safe_path)

    actor = actor_origin.copy(deep=True)
    coa = coa_origin.copy(deep=True)
    safe = safe_origin.copy(deep=True)

    actor = actor.loc[:, ["category", "question", "response", "turn"]]
    coa = coa.loc[:, ["category", "question", "response", "turn"]]
    safe = safe.loc[:, ["category", "question", "response", "turn"]]

    data = pd.concat([actor, coa, safe], axis=0, ignore_index=True)
    data.reset_index(drop=True, inplace=True)

    data["question"] = data["question"].apply(ast.literal_eval)
    data["response"] = data["response"].apply(ast.literal_eval)

    return data

def train_val_test_split(data, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42, type="question"):
    if abs(train_size + val_size + test_size - 1.0) > 1e-7:
        raise ValueError("Train, validation, and test sizes must sum to 1")

    y = data["category"]
    X = data[f"{type}_truncated"]

    remaining_size = train_size + val_size
    test_ratio = test_size / (test_size + remaining_size)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
        shuffle=True
    )

    val_ratio = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state,
        shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    data = loader()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data)
