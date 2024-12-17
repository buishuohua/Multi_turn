# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 11:49
    @ Description:
"""

from dataclasses import dataclass
from config.Model_Settings import ModelSettings
from config.Tokenizer_Settings import TokenizerSettings
from config.Model_Selection import ModelSelection
from config.Training_Settings import TrainingSettings
from config.Data_Settings import DataSettings


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
        """Get default configuration"""
        data_settings = DataSettings(
            which="question",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1
        )
        model_settings = ModelSettings(
            output_dim=38,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.2
        )

        tokenizer_settings = TokenizerSettings(
            name='BERT_base_uncased',
            embedding_type='bert',
            max_length=512,
            truncation='ratio'
        )

        model_selection = ModelSelection(
            model_type='CNNBiLSTM',
            use_cnn=True,
            cnn_layers=2
        )

        training_settings = TrainingSettings(
            num_epochs=100,
            batch_size=32,
            learning_rate=0.001,
            experiment_name='cnn_bilstm_run'
        )

        return cls(
            data_settings=data_settings,
            model_settings=model_settings,
            tokenizer_settings=tokenizer_settings,
            model_selection=model_selection,
            training_settings=training_settings
        )

