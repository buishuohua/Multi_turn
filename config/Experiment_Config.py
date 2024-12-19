# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 11:49
    @ Description:
"""

from dataclasses import dataclass
from config.Data_Settings import DataSettings
from config.Model_Settings import ModelSettings
from config.Tokenizer_Settings import TokenizerSettings
from config.Model_Selection import ModelSelection
from config.Training_Settings import TrainingSettings


@dataclass
class ExperimentConfig:
    """Main configuration class that combines all settings"""
    data_settings: DataSettings
    model_settings: ModelSettings
    tokenizer_settings: TokenizerSettings
    model_selection: ModelSelection
    training_settings: TrainingSettings

    def to_dict(self):
        """Convert all settings to dictionary"""
        return {
            'data_settings': vars(self.data_settings),
            'model_settings': vars(self.model_settings),
            'tokenizer_settings': vars(self.tokenizer_settings),
            'model_selection': vars(self.model_selection),
            'training_settings': vars(self.training_settings)
        }

    @classmethod
    def get_default_config(cls):
        """Get default configuration using component defaults"""
        return cls(
            data_settings=DataSettings.get_default(),
            model_settings=ModelSettings.get_default(),
            tokenizer_settings=TokenizerSettings.get_default(),
            model_selection=ModelSelection.get_default(),
            training_settings=TrainingSettings.get_default()
        )
