# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 10:03
    @ Description:
"""

import os

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import ast
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .truncate import truncate
import torch


def _loader_data():
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


def train_val_test_split(data, train_size, val_size, test_size, type, random_state):
    """Split data preserving both tokenized and truncated columns"""
    if abs(train_size + val_size + test_size - 1.0) > 1e-7:
        raise ValueError("Train, validation, and test sizes must sum to 1")

    y = data["category"]
    # Keep both tokenized and truncated columns
    X = data[[f"{type}_tokenized", f"{type}_truncated"]]

    remaining_size = train_size + val_size
    test_ratio = test_size / (test_size + remaining_size)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=42,
        shuffle=True
    )

    val_ratio = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=42,
        shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def loader(config):
    """Create train, validation and test dataloaders"""
    # Load data
    data = _loader_data()

    # Get tokenizer and apply truncation
    tokenizer = config.tokenizer_settings.get_model().tokenizer

    # Apply truncation first
    data = truncate(
        data=data,
        tokenizer=tokenizer,
        max_tokens=config.tokenizer_settings.max_length,
        which=config.data_settings.which,
        method=config.tokenizer_settings.truncation
    )

    # Convert categories to numbers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['category'])

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        data,
        train_size=config.data_settings.train_size,
        val_size=config.data_settings.val_size,
        test_size=config.data_settings.test_size,
        type=config.data_settings.which,
        random_state=config.data_settings.random_state
    )

    # Create datasets directly from tokenized data
    train_dataset = TensorDataset(
        torch.tensor(X_train[f"{config.data_settings.which}_tokenized"].tolist(), dtype=torch.long),
        torch.tensor(label_encoder.transform(y_train), dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val[f"{config.data_settings.which}_tokenized"].tolist(), dtype=torch.long),
        torch.tensor(label_encoder.transform(y_val), dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test[f"{config.data_settings.which}_tokenized"].tolist(), dtype=torch.long),
        torch.tensor(label_encoder.transform(y_test), dtype=torch.long)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data_settings.batch_size,
        shuffle=config.data_settings.shuffle,
        drop_last=config.data_settings.drop_last
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data_settings.batch_size,
        shuffle=False,
        drop_last=config.data_settings.drop_last
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data_settings.batch_size,
        shuffle=False,
        drop_last=config.data_settings.drop_last
    )

    return train_loader, val_loader, test_loader, label_encoder


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    data = loader()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data)
