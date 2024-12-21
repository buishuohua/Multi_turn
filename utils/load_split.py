# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 10:03
    @ Description:
"""

import os

import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import ast
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .truncate import truncate
import torch
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, ADASYN,
    BorderlineSMOTE, SVMSMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks,
    EditedNearestNeighbours, ClusterCentroids,
    NearMiss, InstanceHardnessThreshold
)
from collections import Counter


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


def handle_imbalanced_data(X, y, config):
    """Handle imbalanced data using the strategy specified in config"""
    strategy = config.data_settings.imbalanced_strategy
    params = config.data_settings.get_strategy_params()

    if strategy == 'none':
        return X, y, None

    if strategy == 'weighted_sampler':
        class_counts = Counter(y)
        weights = {cls: 1.0/count for cls, count in class_counts.items()}
        # Apply alpha parameter to adjust weights
        weights = {cls: w ** params['alpha'] for cls, w in weights.items()}
        sample_weights = torch.DoubleTensor([weights[label] for label in y])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return X, y, sampler

    if strategy == 'random_oversample':
        sampler = RandomOverSampler(
            random_state=config.data_settings.random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy in ['smote', 'borderline1', 'borderline2', 'svm_smote']:
        if strategy == 'smote':
            sampler = SMOTE(
                k_neighbors=params['k_neighbors'],
                sampling_strategy=params['sampling_ratio'],
                random_state=config.data_settings.random_state
            )
        elif strategy in ['borderline1', 'borderline2']:
            sampler = BorderlineSMOTE(
                k_neighbors=params['k_neighbors'],
                sampling_strategy=params['sampling_ratio'],
                random_state=config.data_settings.random_state,
                kind=params['kind']
            )
        else:  # svm_smote
            sampler = SVMSMOTE(
                k_neighbors=params['k_neighbors'],
                sampling_strategy=params['sampling_ratio'],
                random_state=config.data_settings.random_state,
                svm_estimator=params['svm_estimator']
            )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy == 'adasyn':
        sampler = ADASYN(
            n_neighbors=params['n_neighbors'],
            sampling_strategy=params['sampling_ratio'],
            random_state=config.data_settings.random_state
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy == 'random_undersample':
        sampler = RandomUnderSampler(
            random_state=config.data_settings.random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy == 'tomek':
        sampler = TomekLinks(sampling_strategy=params['sampling_strategy'])
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy == 'edited_nearest_neighbors':
        sampler = EditedNearestNeighbours(
            n_neighbors=params['n_neighbors'],
            kind_sel=params['kind_sel']
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy == 'cluster_centroids':
        sampler = ClusterCentroids(
            estimator=params['estimator'],
            voting=params['voting'],
            random_state=config.data_settings.random_state
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy == 'near_miss':
        sampler = NearMiss(
            version=params['version'],
            n_neighbors=params['n_neighbors']
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    if strategy == 'instance_hardness_threshold':
        sampler = InstanceHardnessThreshold(
            estimator=params['estimator'],
            cv=params['cv'],
            threshold=params['threshold'],
            random_state=config.data_settings.random_state
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None

    raise ValueError(f"Unknown imbalanced strategy: {strategy}")


def train_val_test_split(data, config):
    y = data["category"]
    X = data[[f"{config.data_settings.which}_tokenized",
              f"{config.data_settings.which}_truncated"]]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.data_settings.test_size,
        stratify=y,
        random_state=config.data_settings.random_state
    )

    val_ratio = config.data_settings.val_size / \
        (config.data_settings.train_size + config.data_settings.val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=config.data_settings.random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def loader(config):
    data = _loader_data()

    tokenizer = config.tokenizer_settings.get_model().tokenizer
    data = truncate(
        data=data,
        tokenizer=tokenizer,
        max_tokens=config.tokenizer_settings.max_length,
        which=config.data_settings.which,
        method=config.tokenizer_settings.truncation
    )

    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        data, config)

    # Convert pandas Series to numpy arrays
    X_train_tokenized = np.array(
        X_train[f"{config.data_settings.which}_tokenized"].tolist())
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Handle imbalanced data
    X_train_resampled, y_train_resampled, sampler = handle_imbalanced_data(
        X_train_tokenized, y_train, config
    )

    # Create datasets with numpy arrays
    train_dataset = TensorDataset(
        torch.tensor(X_train_resampled, dtype=torch.long),
        torch.tensor(y_train_resampled, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(
            X_val[f"{config.data_settings.which}_tokenized"].tolist(), dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(
            X_test[f"{config.data_settings.which}_tokenized"].tolist(), dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training_settings.batch_size,
        shuffle=config.data_settings.shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=config.data_settings.drop_last
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training_settings.batch_size,
        shuffle=False,
        drop_last=config.data_settings.drop_last
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training_settings.batch_size,
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
