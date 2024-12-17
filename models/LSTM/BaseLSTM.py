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

from config.Model_Settings import ModelSettings
from models.Embedding.BERT_tokenizer import BERTs


class BaseLSTM(nn.Module, ABC):
    """
    Base class for LSTM-based architectures.
    Inherit from this class to create specific LSTM variants.
    """

    def __init__(self, settings: ModelSettings):
        super().__init__()
        self.settings = settings

        # Common layers that most LSTM variants will use
        self.embedding = self._create_embedding()
        self.lstm = self._create_lstm()
        self.fc_layers = self._create_fc_layers()

    def _create_embedding(self) -> nn.Module:
        """Create embedding layer"""
        return BERTs.get_tokenizer(self.settings.embedding_type).tokenizer

    def _create_lstm(self) -> nn.Module:
        """Create LSTM layer"""
        return nn.LSTM(
            input_size=self.settings.hidden_dims[0],
            hidden_size=self.settings.hidden_dims[0],
            num_layers=len(self.settings.hidden_dims),
            batch_first=True,
            bidirectional=True,
            dropout=self.settings.dropout_rate if len(self.settings.hidden_dims) > 1 else 0
        )

    def _create_fc_layers(self) -> nn.Module:
        """Create fully connected layers"""
        layers = []
        prev_dim = self.settings.hidden_dims[0] * 2  # * 2 for bidirectional

        for hidden_dim in self.settings.hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.settings.dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.settings.output_dim))
        return nn.Sequential(*layers)

    def _get_activation(self) -> nn.Module:
        """Get activation function based on settings"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(self.settings.activation.lower(), nn.ReLU())

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states"""
        weight = next(self.parameters())
        h0 = weight.new_zeros(
            2 * len(self.settings.hidden_dims),  # * 2 for bidirectional
            batch_size,
            self.settings.hidden_dims[0]
        )
        c0 = weight.new_zeros(
            2 * len(self.settings.hidden_dims),
            batch_size,
            self.settings.hidden_dims[0]
        )
        return h0, c0

    @abstractmethod
    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass to be implemented by child classes

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden: Optional initial hidden states

        Returns:
            output: Model output
            (h_n, c_n): Final hidden state and cell state
        """
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.settings.learning_rate)

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        return {
            'model_type': self.__class__.__name__,
            'input_dim': self.settings.input_dim,
            'output_dim': self.settings.output_dim,
            'hidden_dims': self.settings.hidden_dims,
            'dropout_rate': self.settings.dropout_rate,
            'activation': self.settings.activation
        }
