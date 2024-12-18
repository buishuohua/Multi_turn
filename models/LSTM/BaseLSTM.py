# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 12:57
    @ Description:
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

from config.Experiment_Config import ExperimentConfig


class BaseLSTM(nn.Module, ABC):
    """
    Base class for LSTM-based architectures.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        # Get embedding model
        self.embedding_model = config.tokenizer_settings.get_model().model
        if not config.model_settings.fine_tune_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

        # Common layers
        self.lstm = self._create_lstm()
        self.fc_layers = self._create_fc_layers()
        self.dropout = nn.Dropout(config.model_settings.dropout_rate)
        self.final_activation = config.model_settings.get_final_activation()

    def _create_lstm(self) -> nn.Module:
        """Create LSTM layer"""
        return nn.LSTM(
            input_size=self.config.model_settings.embedding_dim,
            hidden_size=self.config.model_settings.hidden_dims[0],
            num_layers=self.config.model_settings.num_layers,
            batch_first=True,
            bidirectional=self.config.model_settings.bidirectional,
            dropout=self.config.model_settings.dropout_rate if self.config.model_settings.num_layers > 1 else 0
        )

    def _create_fc_layers(self) -> nn.Module:
        """Create fully connected layers"""
        layers = []
        prev_dim = self.config.model_settings.hidden_dims[0] * (
            2 if self.config.model_settings.bidirectional else 1)

        for hidden_dim in self.config.model_settings.hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.config.model_settings.get_activation(),
                nn.Dropout(self.config.model_settings.dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(
            nn.Linear(prev_dim, self.config.model_settings.output_dim))
        return nn.Sequential(*layers)

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states"""
        num_directions = 2 if self.config.model_settings.bidirectional else 1
        weight = next(self.parameters())
        h0 = weight.new_zeros(
            self.config.model_settings.num_layers * num_directions,
            batch_size,
            self.config.model_settings.hidden_dims[0]
        )
        c0 = weight.new_zeros(
            self.config.model_settings.num_layers * num_directions,
            batch_size,
            self.config.model_settings.hidden_dims[0]
        )
        return h0, c0

    @abstractmethod
    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        return {
            'model_type': self.__class__.__name__,
            'embedding_type': self.config.model_settings.embedding_type,
            'embedding_dim': self.config.model_settings.embedding_dim,
            'hidden_dims': self.config.model_settings.hidden_dims,
            'num_layers': self.config.model_settings.num_layers,
            'bidirectional': self.config.model_settings.bidirectional,
            'dropout_rate': self.config.model_settings.dropout_rate,
            'activation': self.config.model_settings.activation,
            'output_activation': self.config.model_settings.final_activation,
            'fine_tune_embedding': self.config.model_settings.fine_tune_embedding,
            'loss': self.config.model_settings.loss
        }
