# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/16 12:14
    @ Description:
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DataSettings:
    """Data settings with imbalanced data handling strategies"""

    # Basic data settings
    which: str = "question"
    random_state: int = 42
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    shuffle: bool = True
    drop_last: bool = True

    # Imbalanced strategy name (main control)
    imbalanced_strategy: Literal[
        'none',
        'weighted_sampler',
        'class_weights',
        'random_oversample',
        'smote',
        'adasyn',
        'borderline1',
        'borderline2',
        'svm_smote',
        'random_undersample',
        'tomek',
        'edited_nearest_neighbors',
        'cluster_centroids',
        'near_miss',
        'instance_hardness_threshold'
    ] = 'none'

    # Weighted sampler parameters
    weighted_sampler_alpha: float = 1.0  # Weight adjustment factor

    # Class weights parameters
    class_weight_method: Literal['balanced', 'balanced_subsample'] = 'balanced'

    # SMOTE and variants parameters
    smote_k_neighbors: int = 5
    # 1.0 means balance all classes to majority class
    smote_sampling_ratio: float = 1.0
    svm_smote_svm_estimator: Optional[str] = None
    borderline_kind: Literal['borderline-1', 'borderline-2'] = 'borderline-1'

    # ADASYN parameters
    adasyn_n_neighbors: int = 5
    adasyn_sampling_ratio: float = 1.0

    # Undersampling parameters
    # NearMiss parameters
    near_miss_version: int = 1  # 1, 2, or 3
    near_miss_n_neighbors: int = 3

    # Edited Nearest Neighbors parameters
    enn_n_neighbors: int = 3
    enn_kind_sel: Literal['all', 'mode'] = 'all'

    # Cluster Centroids parameters
    cluster_centroids_estimator: str = 'kmeans'
    cluster_centroids_voting: Literal['auto', 'hard', 'soft'] = 'auto'

    # Instance Hardness Threshold parameters
    iht_estimator: str = 'random_forest'
    iht_cv: int = 5
    iht_threshold: float = 0.3

    # Tomek parameters
    tomek_sampling_strategy: Literal['auto', 'majority',
                                     'not minority', 'not majority', 'all'] = 'auto'

    def __post_init__(self):
        """Validate and adjust settings after initialization"""
        # Validate split sizes
        total_split = self.train_size + self.val_size + self.test_size
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Split sizes must sum to 1.0, got {total_split}")

        # Set strategy flags based on imbalanced_strategy
        self.use_weighted_sampler = self.imbalanced_strategy == 'weighted_sampler'
        self.use_class_weights = self.imbalanced_strategy == 'class_weights'
        self.oversample = self.imbalanced_strategy in {
            'random_oversample', 'smote', 'adasyn',
            'borderline1', 'borderline2', 'svm_smote'
        }
        self.undersample = self.imbalanced_strategy in {
            'random_undersample', 'tomek', 'edited_nearest_neighbors',
            'cluster_centroids', 'near_miss', 'instance_hardness_threshold'
        }

        # Validate strategy-specific parameters
        self._validate_strategy_parameters()

    def _validate_strategy_parameters(self):
        """Validate parameters specific to each strategy"""
        if self.imbalanced_strategy == 'weighted_sampler':
            if self.weighted_sampler_alpha <= 0:
                raise ValueError("weighted_sampler_alpha must be positive")

        elif self.imbalanced_strategy in {'smote', 'borderline1', 'borderline2', 'svm_smote'}:
            if self.smote_k_neighbors < 1:
                raise ValueError("smote_k_neighbors must be >= 1")
            if not 0 < self.smote_sampling_ratio <= 1.0:
                raise ValueError(
                    "smote_sampling_ratio must be between 0 and 1")

        elif self.imbalanced_strategy == 'adasyn':
            if self.adasyn_n_neighbors < 1:
                raise ValueError("adasyn_n_neighbors must be >= 1")
            if not 0 < self.adasyn_sampling_ratio <= 1.0:
                raise ValueError(
                    "adasyn_sampling_ratio must be between 0 and 1")

        elif self.imbalanced_strategy == 'near_miss':
            if self.near_miss_version not in {1, 2, 3}:
                raise ValueError("near_miss_version must be 1, 2, or 3")
            if self.near_miss_n_neighbors < 1:
                raise ValueError("near_miss_n_neighbors must be >= 1")

        elif self.imbalanced_strategy == 'edited_nearest_neighbors':
            if self.enn_n_neighbors < 1:
                raise ValueError("enn_n_neighbors must be >= 1")

        elif self.imbalanced_strategy == 'instance_hardness_threshold':
            if not 0 <= self.iht_threshold <= 1:
                raise ValueError("iht_threshold must be between 0 and 1")
            if self.iht_cv < 2:
                raise ValueError("iht_cv must be >= 2")

    @classmethod
    def get_default(cls):
        """Return default settings"""
        return cls()

    def get_active_strategy(self) -> str:
        """Return the active imbalanced data handling strategy"""
        return self.imbalanced_strategy

    def get_strategy_params(self) -> dict:
        """Return parameters specific to the active strategy"""
        params = {}

        if self.imbalanced_strategy == 'weighted_sampler':
            params['alpha'] = self.weighted_sampler_alpha

        elif self.imbalanced_strategy == 'class_weights':
            params['class_weight'] = self.class_weight_method

        elif 'smote' in self.imbalanced_strategy:
            params.update({
                'k_neighbors': self.smote_k_neighbors,
                'sampling_ratio': self.smote_sampling_ratio
            })
            if self.imbalanced_strategy == 'svm_smote':
                params['svm_estimator'] = self.svm_smote_svm_estimator
            elif self.imbalanced_strategy in {'borderline1', 'borderline2'}:
                params['kind'] = self.borderline_kind

        elif self.imbalanced_strategy == 'adasyn':
            params.update({
                'n_neighbors': self.adasyn_n_neighbors,
                'sampling_ratio': self.adasyn_sampling_ratio
            })

        elif self.imbalanced_strategy == 'near_miss':
            params.update({
                'version': self.near_miss_version,
                'n_neighbors': self.near_miss_n_neighbors
            })

        elif self.imbalanced_strategy == 'edited_nearest_neighbors':
            params.update({
                'n_neighbors': self.enn_n_neighbors,
                'kind_sel': self.enn_kind_sel
            })

        elif self.imbalanced_strategy == 'cluster_centroids':
            params.update({
                'estimator': self.cluster_centroids_estimator,
                'voting': self.cluster_centroids_voting
            })

        elif self.imbalanced_strategy == 'instance_hardness_threshold':
            params.update({
                'estimator': self.iht_estimator,
                'cv': self.iht_cv,
                'threshold': self.iht_threshold
            })

        elif self.imbalanced_strategy == 'tomek':
            params['sampling_strategy'] = self.tomek_sampling_strategy

        return params
