# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 12:57
    @ Description:
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from config.Experiment_Config import ExperimentConfig

# Add this import for runtime
from typing import Any as ExperimentConfig  # temporary type alias


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
        self.final_activation = config.model_settings.get_final_activation()
        self.dropout = nn.Dropout(config.model_settings.dropout_rate)
        self.lstm = self._create_lstm()
        self.fc_layers = self._create_fc_layers()

        # Initialize weights after creating layers
        self._initialize_weights()

    def _create_lstm(self) -> nn.Module:
        """Create LSTM layer with different hidden sizes for each layer"""
        # Create a list of LSTM layers with different hidden sizes
        lstm_layers = []
        input_size = self.config.model_settings.embedding_dim

        for i, hidden_size in enumerate(self.config.model_settings.hidden_dims):
            lstm_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,  # Each LSTM layer has 1 layer
                batch_first=True,
                bidirectional=self.config.model_settings.bidirectional,
                dropout=0  # Dropout will be handled between layers
            )
            lstm_layers.append(lstm_layer)
            # Next layer's input size is current hidden_size * 2 if bidirectional
            input_size = hidden_size * \
                (2 if self.config.model_settings.bidirectional else 1)

        return nn.ModuleList(lstm_layers)

    def _create_fc_layers(self) -> nn.Module:
        """Create fully connected layers"""
        # Use the last LSTM layer's hidden size
        last_hidden_size = self.config.model_settings.hidden_dims[-1] * (
            2 if self.config.model_settings.bidirectional else 1)

        return nn.Sequential(
            nn.Linear(last_hidden_size, self.config.model_settings.output_dim),
            self.final_activation if self.final_activation is not None else nn.Identity()
        )

    def init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for each LSTM layer"""
        hidden_states = []
        for i, hidden_size in enumerate(self.config.model_settings.hidden_dims):
            num_directions = 2 if self.config.model_settings.bidirectional else 1
            weight = next(self.parameters())
            h0 = weight.new_zeros(num_directions, batch_size, hidden_size)
            c0 = weight.new_zeros(num_directions, batch_size, hidden_size)
            hidden_states.append((h0, c0))
        return hidden_states

    @abstractmethod
    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        # Add initialization info
        info.update({
            'weight_initialization': self.config.model_settings.weight_init,
            'init_gain': self.config.model_settings.init_gain,
            'init_std': self.config.model_settings.init_std
        })
        return info

    def _initialize_weights(self):
        """Initialize weights based on configuration"""
        init_config = self.config.model_settings.get_initializer()

        def init_layer(layer):
            if isinstance(layer, (nn.Linear, nn.LSTM)):
                # Initialize weights
                if isinstance(layer, nn.LSTM):
                    weights = [layer.weight_ih_l0, layer.weight_hh_l0]
                    if layer.bidirectional:
                        weights.extend(
                            [layer.weight_ih_l0_reverse, layer.weight_hh_l0_reverse])
                else:
                    weights = [layer.weight]

                for weight in weights:
                    if init_config['method'] == 'xavier_uniform':
                        nn.init.xavier_uniform_(
                            weight, gain=init_config['params']['gain'])
                    elif init_config['method'] == 'xavier_normal':
                        nn.init.xavier_normal_(
                            weight, gain=init_config['params']['gain'])
                    elif init_config['method'] == 'kaiming_uniform':
                        nn.init.kaiming_uniform_(
                            weight, a=init_config['params']['a'])
                    elif init_config['method'] == 'kaiming_normal':
                        nn.init.kaiming_normal_(
                            weight, a=init_config['params']['a'])
                    elif init_config['method'] == 'orthogonal':
                        nn.init.orthogonal_(
                            weight, gain=init_config['params']['gain'])
                    elif init_config['method'] == 'normal':
                        nn.init.normal_(weight,
                                        mean=init_config['params']['mean'],
                                        std=init_config['params']['std'])
                    elif init_config['method'] == 'uniform':
                        nn.init.uniform_(weight,
                                         a=init_config['params']['a'],
                                         b=init_config['params']['b'])

                # Initialize biases
                if isinstance(layer, nn.LSTM):
                    biases = [layer.bias_ih_l0, layer.bias_hh_l0]
                    if layer.bidirectional:
                        biases.extend([layer.bias_ih_l0_reverse,
                                      layer.bias_hh_l0_reverse])
                else:
                    biases = [layer.bias] if layer.bias is not None else []

                for bias in biases:
                    nn.init.zeros_(bias)

        # Apply initialization to all layers
        for lstm in self.lstm:
            init_layer(lstm)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                init_layer(layer)
