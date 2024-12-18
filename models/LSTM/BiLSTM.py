# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 14:31
    @ Description:
"""

from .BaseLSTM import BaseLSTM
from config.Experiment_Config import ExperimentConfig
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class BiLSTM(BaseLSTM):
    def __init__(self, config: ExperimentConfig):
        """
        Bidirectional LSTM implementation

        Args:
            config: Complete experiment configuration including model and training settings
        """
        super().__init__(config)
        self.dropout = nn.Dropout(config.model_settings.dropout_rate)

    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass implementation

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional initial hidden states

        Returns:
            output: Model output
            (h_n, c_n): Final hidden state and cell state
        """

        batch_size = x.size(0)

        # Get embeddings
        if not self.config.model_settings.fine_tune_embedding:
            with torch.no_grad():
                embeddings = self.embedding_model(x)[0]
        else:
            embeddings = self.embedding_model(x)[0]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embeddings, hidden)

        # We need the last time step output for classification
        # Take the last time step or mean of all time steps
        if self.config.model_settings.pooling == 'last':
            lstm_out = lstm_out[:, -1, :]  # Take last time step
        else:
            lstm_out = torch.mean(lstm_out, dim=1)  # Mean pooling

        # Apply dropout and FC layers
        lstm_out = self.dropout(lstm_out)
        output = self.fc_layers(lstm_out)

        return output, (h_n, c_n)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        info.update({
            'model_variant': 'BiLSTM',
            'bidirectional': True,
            'device': self.config.training_settings.device
        })
        return info
