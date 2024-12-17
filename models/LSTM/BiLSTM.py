# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 14:31
    @ Description:
"""

from .BaseLSTM import BaseLSTM
from config.Model_Settings import ModelSettings
import torch
from typing import Tuple, Optional, Dict, Any

class BiLSTM(BaseLSTM):
    def __init__(self, settings: ModelSettings):
        """
        Bidirectional LSTM implementation

        Args:
            settings (ModelSettings): Model configuration settings
        """
        super().__init__(settings)

        # Additional initialization if needed for BiLSTM specific features
        self.dropout = nn.Dropout(settings.dropout_rate)

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

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # BiLSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(embeddings, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Pass through fully connected layers
        output = self.fc_layers(lstm_out)

        return output, (h_n, c_n)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        info.update({
            'model_variant': 'BiLSTM',
            'embedding_type': self.settings.embedding_type,
            'bidirectional': True,
            'num_layers': len(self.settings.hidden_dims)
        })
        return info
