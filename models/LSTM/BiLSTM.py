# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 14:31
    @ Description:
"""

from .BaseLSTM import BaseLSTM
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
from config.Experiment_Config import ExperimentConfig


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
                hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[
            torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass implementation

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional initial hidden states for each layer

        Returns:
            output: Model output
            hidden_states: List of (h_n, c_n) for each layer
        """
        batch_size = x.size(0)

        # Get embeddings
        if not self.config.model_settings.fine_tune_embedding:
            with torch.no_grad():
                x = self.embedding_model(x)[0]
        else:
            x = self.embedding_model(x)[0]

        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # Process through LSTM layers
        current_hidden_states = []
        for i, lstm_layer in enumerate(self.lstm):
            # Process current layer
            x, (h_n, c_n) = lstm_layer(x, hidden[i])
            current_hidden_states.append((h_n, c_n))

            # Apply dropout between layers (except last layer)
            if i < len(self.lstm) - 1:
                x = self.dropout(x)

        # Apply pooling strategy
        if self.config.model_settings.pooling == 'last':
            # Take the last time step
            x = x[:, -1, :]
        else:  # 'mean' pooling
            # Average across all time steps
            x = torch.mean(x, dim=1)

        # Final dropout before FC layer
        x = self.dropout(x)

        # Pass through fully connected layers
        output = self.fc_layers(x)

        return output, current_hidden_states

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        info.update({
            'model_type': 'BiLSTM',
            'bidirectional': self.config.model_settings.bidirectional,
            'num_layers': len(self.lstm),
            'hidden_dims': self.config.model_settings.hidden_dims,
            'pooling_strategy': self.config.model_settings.pooling,
            'dropout_rate': self.config.model_settings.dropout_rate
        })
        return info
