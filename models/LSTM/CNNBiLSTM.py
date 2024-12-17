# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 11:17
    @ Description:
"""
from .BaseLSTM import BaseLSTM
from config.Model_Settings import ModelSettings
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class CNNBiLSTM(BaseLSTM):
    def __init__(self, settings: ModelSettings):
        """
        CNN + Bidirectional LSTM implementation

        Args:
            settings (ModelSettings): Model configuration settings
        """
        super().__init__(settings)

        # CNN layers
        self.conv_layers = self._create_conv_layers()
        self.dropout = nn.Dropout(settings.dropout_rate)

    def _create_conv_layers(self) -> nn.Module:
        """Create CNN layers for feature extraction"""
        return nn.Sequential(
            # First conv layer
            nn.Conv1d(
                in_channels=self.settings.hidden_dims[0],  # Input channels (BERT embedding dim)
                out_channels=self.settings.hidden_dims[0],  # Keep same dimension
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Dropout(self.settings.dropout_rate),

            # Second conv layer
            nn.Conv1d(
                in_channels=self.settings.hidden_dims[0],
                out_channels=self.settings.hidden_dims[0],
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Dropout(self.settings.dropout_rate)
        )

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

        # Get embeddings using BERT tokenizer
        embeddings = self.embedding(x)

        # CNN expects (batch, channels, length) format
        # Current shape: (batch, seq_len, embedding_dim)
        # Need to transpose to: (batch, embedding_dim, seq_len)
        conv_input = embeddings.transpose(1, 2)

        # Apply CNN layers
        conv_output = self.conv_layers(conv_input)

        # Transpose back for LSTM
        # From: (batch, embedding_dim, seq_len)
        # To: (batch, seq_len, embedding_dim)
        lstm_input = conv_output.transpose(1, 2)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # BiLSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(lstm_input, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Pass through fully connected layers
        output = self.fc_layers(lstm_out)

        return output, (h_n, c_n)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        info.update({
            'model_variant': 'CNNBiLSTM',
            'embedding_type': self.settings.embedding_type,
            'bidirectional': True,
            'num_layers': len(self.settings.hidden_dims),
            'cnn_layers': 2,
            'kernel_size': 3
        })
        return info
