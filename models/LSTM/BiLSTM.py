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
import torch.nn.functional as F


class BiLSTM(BaseLSTM):
    """
    Bidirectional LSTM with attention mechanism.

    Features:
    - Multiple LSTM layers with decreasing hidden sizes
    - Attention mechanism for sequence representation
    - Dropout regularization
    """

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
        Forward pass implementation for BiLSTM

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional initial hidden states for each layer

        Returns:
            output: Model output of shape (batch_size, num_classes)
            hidden_states: List of final hidden states for each layer
        """
        batch_size = x.size(0)

        # Get embeddings
        if not self.config.model_settings.fine_tune_embedding:
            with torch.no_grad():
                # shape: [batch_size, seq_len, embedding_dim]
                x = self.embedding_model(x)[0]
        else:
            x = self.embedding_model(x)[0]

        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # Process through LSTM layers
        current_hidden_states = []
        residual = x  # Store input for residual connection

        for i, lstm_layer in enumerate(self.lstm):
            # Store the input for residual connection
            residual = x

            # Process through LSTM
            lstm_out, (h_n, c_n) = lstm_layer(x, hidden[i])
            current_hidden_states.append((h_n, c_n))

            # Apply layer normalization if enabled
            if self.layer_norms is not None:
                lstm_out = self.layer_norms[i](lstm_out)

            # Apply residual connection if enabled
            if self.config.model_settings.use_res_net:
                projected_residual = self.res_projections[i](residual)
                lstm_out = lstm_out + self.res_dropout(projected_residual)

            if i < len(self.lstm) - 1:
                lstm_out = self.dropout(lstm_out)

            x = lstm_out

        # Apply attention if configured
        if self.attention is not None:
            x, _ = self.attention(x)

        # Apply pooling
        if self.config.model_settings.pooling == 'mean':
            x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        else:  # 'last'
            x = x[:, -1, :]  # [batch_size, hidden_dim]

        # Final dropout and FC layers
        x = self.dropout(x)
        logits = self.fc_layers(x)  # [batch_size, num_classes]

        return logits, current_hidden_states

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        info.update({
            'model_type': 'BiLSTM',
            'bidirectional': self.config.model_settings.bidirectional,
            'num_layers': len(self.lstm)
        })
        return info
