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


def length_process(df, which):
    max_len = max(df["turn"])

    for i in range(max_len):
        col_name = f'len_{i}_{which}'
        df[col_name] = df[f"len_lst_{which}"].apply(lambda x: x[i] if i < len(x) else None)

    return df


def loader(type="ques"):
    pwd = os.getcwd()
    actor_file = "./data/gather_actor_attack_results.csv"
    coa_file = "./data/gather_coa_attack_results.csv"
    safe_file = "./data/gather_safe_results.csv"
    actor_path = os.path.join(pwd, actor_file)
    coa_path = os.path.join(pwd, coa_file)
    safe_path = os.path.join(pwd, safe_file)

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

    data["len_lst_ques"] = data["question"].apply(lambda x: [len(item) for item in x])
    data["len_lst_resp"] = data["response"].apply(lambda x: [len(item) for item in x])

    # data = length_process(data, "ques")
    # data = length_process(data, "resp")

    if type == "ques":
        return data.drop(columns=["response"])
    elif type == "resp":
        return data.drop(columns=["question"])
    else:
        raise TypeError("type must be 'ques' or 'resp'")


def train_val_test_split(data, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    if abs(train_size + val_size + test_size - 1.0) > 1e-7:
        raise ValueError("Train, validation, and test sizes must sum to 1")

    y = data["category"]
    X = data.drop(columns=["category"])

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
