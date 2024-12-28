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

        # Create attention modules
        self.attention_modules = self._create_attention()

        # Ensure hidden_dims is properly initialized
        if not hasattr(self.config.model_settings, 'hidden_dims') or not self.config.model_settings.hidden_dims:
            raise ValueError(
                "hidden_dims must be initialized before creating layer norms")

        # Initialize layer norms with consistent naming
        if self.config.model_settings.use_layer_norm:
            # Create layer norms for each LSTM layer
            lstm_layer_norms = {}
            for i in range(len(self.lstm)):  # Use len(self.lstm) instead of num_layers
                layer_dim = self.config.model_settings.hidden_dims[i] * (
                    2 if config.model_settings.bidirectional else 1)
                lstm_layer_norms[f'lstm_{i}'] = nn.LayerNorm(
                    layer_dim,
                    eps=config.model_settings.layer_norm_eps,
                    elementwise_affine=config.model_settings.layer_norm_affine
                )

            # Add final layer norm
            final_dim = self.config.model_settings.hidden_dims[-1] * (
                2 if config.model_settings.bidirectional else 1)
            lstm_layer_norms['final'] = nn.LayerNorm(
                final_dim,
                eps=config.model_settings.layer_norm_eps,
                elementwise_affine=config.model_settings.layer_norm_affine
            )

            self.layer_norms = nn.ModuleDict(lstm_layer_norms)

    def forward(self, x):
        # Get embeddings
        # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding_model(x).last_hidden_state

        # Apply attention after embedding
        if self.attention_modules.get('embedding'):
            embedded = self.apply_attention(
                embedded, self.attention_modules['embedding'])

        # Process through LSTM layers with attention
        lstm_out = embedded
        for i, lstm_layer in enumerate(self.lstm):
            # Store residual
            residual = lstm_out

            # LSTM forward pass - let PyTorch handle hidden state initialization
            lstm_out, _ = lstm_layer(lstm_out)

            # Apply attention between LSTM layers
            if i < len(self.lstm) - 1 and self.attention_modules.get('inter_lstm'):
                lstm_out = self.apply_attention(lstm_out,
                                                self.attention_modules['inter_lstm'][i])

            # Apply residual connection if enabled
            if self.config.model_settings.use_res_net:
                projected_residual = self.res_projections[i](residual)
                if self.res_dropout is not None:
                    projected_residual = self.res_dropout(projected_residual)
                lstm_out = lstm_out + projected_residual

            # Apply layer normalization if enabled
            if self.layer_norms is not None:
                lstm_out = self.layer_norms[f'lstm_{i}'](lstm_out)

        # Get final hidden state based on pooling style
        if self.config.model_settings.pooling == 'mean':
            # Mean pooling over sequence length
            hidden = lstm_out.mean(dim=1)
        else:  # 'last' pooling (default)
            if self.config.model_settings.bidirectional:
                # Concatenate forward (last timestep) and backward (first timestep) hidden states
                hidden = torch.cat((lstm_out[:, -1, :self.config.model_settings.hidden_dims[-1]],
                                    lstm_out[:, 0, self.config.model_settings.hidden_dims[-1]:]), dim=1)
            else:
                # Use last timestep for unidirectional LSTM
                hidden = lstm_out[:, -1, :]

        # Apply attention after final LSTM
        if self.attention_modules.get('output'):
            hidden = self.apply_attention(hidden.unsqueeze(1),
                                          self.attention_modules['output']).squeeze(1)

        # Apply final layer norm if enabled
        if self.layer_norms is not None:
            hidden = self.layer_norms['final'](hidden)

        # Store activations for monitoring
        activations = {
            'first_layer': hidden.detach(),
            'middle_layer': None,
            'last_layer': None
        }

        # Pass through FC layers
        x = self.fc_layers(hidden)
        activations['last_layer'] = x.detach()

        return x, activations

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        info.update({
            'model_type': 'BiLSTM',
            'bidirectional': self.config.model_settings.bidirectional,
            'num_layers': len(self.lstm)
        })
        return info
