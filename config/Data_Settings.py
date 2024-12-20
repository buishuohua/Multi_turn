# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/16 12:14
    @ Description:
"""

from dataclasses import dataclass, field
from typing import Literal


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

    # Imbalanced data handling flags
    use_weighted_sampler: bool = False
    use_class_weights: bool = False
    oversample: bool = False
    undersample: bool = False

    # Strategy settings
    oversample_strategy: Literal[
        'random',
        'smote',
        'adasyn',
        'borderline1',
        'borderline2',
        'svm_smote'
    ] = 'random'

    undersample_strategy: Literal[
        'random',
        'tomek',
        'edited_nearest_neighbors',
        'cluster_centroids',
        'near_miss',
        'instance_hardness_threshold'
    ] = 'random'

    # SMOTE specific parameters
    smote_k_neighbors: int = 5
    # 1.0 means balance all classes to majority class
    smote_sampling_ratio: float = 1.0

    def __post_init__(self):
        """Validate and adjust settings after initialization"""
        # Validate split sizes
        total_split = self.train_size + self.val_size + self.test_size
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Split sizes must sum to 1.0, got {total_split}")

        # Ensure only one sampling strategy is active
        active_strategies = sum([
            self.use_weighted_sampler,
            self.use_class_weights,
            self.oversample,
            self.undersample
        ])

        if active_strategies > 1:
            # Priority order: weighted_sampler > class_weights > oversample > undersample
            if self.use_weighted_sampler:
                self.use_class_weights = False
                self.oversample = False
                self.undersample = False
            elif self.use_class_weights:
                self.oversample = False
                self.undersample = False
            elif self.oversample:
                self.undersample = False

            print("Warning: Multiple imbalanced data strategies detected. "
                  "Using priority order: weighted_sampler > class_weights > "
                  "oversample > undersample")

        # Validate SMOTE parameters
        if self.oversample and self.oversample_strategy.startswith('smote'):
            if self.smote_k_neighbors < 1:
                raise ValueError("SMOTE k_neighbors must be >= 1")
            if not 0 < self.smote_sampling_ratio <= 1.0:
                raise ValueError(
                    "SMOTE sampling ratio must be between 0 and 1")

    @classmethod
    def get_default(cls):
        """Return default settings"""
        return cls(
            which="question",
            random_state=42,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            shuffle=True,
            drop_last=True,
            use_weighted_sampler=False,
            use_class_weights=False,
            oversample=False,
            undersample=False,
            oversample_strategy='random',
            undersample_strategy='random',
            smote_k_neighbors=5,
            smote_sampling_ratio=1.0
        )

    def get_active_strategy(self) -> str:
        """Return the active imbalanced data handling strategy"""
        if self.use_weighted_sampler:
            return "weighted_sampler"
        elif self.use_class_weights:
            return "class_weights"
        elif self.oversample:
            return f"oversample_{self.oversample_strategy}"
        elif self.undersample:
            return f"undersample_{self.undersample_strategy}"
        else:
            return "none"
