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
import math
import torch.nn.functional as F
import os
import re

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
        self.embedding_model = self._load_or_create_embedding_model()
        if not config.model_settings.fine_tune_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

        # Common layers
        self.final_activation = config.model_settings.get_final_activation()
        self.dropout = nn.Dropout(config.model_settings.dropout_rate)
        self.res_dropout = nn.Dropout(
            config.model_settings.res_dropout) if config.model_settings.use_res_net else None
        self.lstm = self._create_lstm()

        # Create projection layers for residual connections if dimensions don't match
        self.res_projections = self._create_res_projections(
        ) if config.model_settings.use_res_net else None

        # Add attention layer if enabled
        self.attention = None
        if config.model_settings.use_attention:
            self.attention = self._create_attention()

        self.fc_layers = self._create_fc_layers()

        # Create layer normalization layers if enabled
        self.layer_norms = self._create_layer_norms(
        ) if config.model_settings.use_layer_norm else None

        # Initialize weights after creating layers
        self._initialize_weights()

    def _load_or_create_embedding_model(self):
        """Load fine-tuned embedding model if exists, otherwise create new"""
        if self.config.model_settings.fine_tune_embedding:
            # Get the base model
            embedding_model = self.config.tokenizer_settings.get_model().model

            # For full fine-tuning, enable gradients for all parameters
            for param in embedding_model.parameters():
                param.requires_grad = True

            print("\n[Fine-tuning] Enabled full fine-tuning of embedding model")

            # Try to load existing fine-tuned model if it exists
            fine_tuned_dir = os.path.join(
                self.config.training_settings.fine_tuned_models_dir,
                self.config.model_settings.embedding_type
            )

            if os.path.exists(fine_tuned_dir):
                # Try loading best model first
                best_path = os.path.join(
                    fine_tuned_dir, 'best', 'embedding_model_best.pt')
                if os.path.exists(best_path):
                    try:
                        checkpoint = torch.load(best_path)
                        embedding_model.load_state_dict(
                            checkpoint['state_dict'])
                        print(f"✅ Loaded best fine-tuned embedding model")
                        return embedding_model
                    except Exception as e:
                        print(f"⚠️ Could not load best model: {str(e)}")

            print("🆕 Starting with fresh embedding model for fine-tuning")
            return embedding_model

        # If not fine-tuning, return the model with frozen parameters
        model = self.config.tokenizer_settings.get_model().model
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _create_attention(self) -> nn.Module:
        """Create attention module based on configuration"""
        if not self.config.model_settings.use_attention:
            return None

        from models.Attention.multi_head_attention import MultiHeadAttention
        return MultiHeadAttention(self.config)

    def _create_lstm(self) -> nn.Module:
        """Create LSTM layer with different hidden sizes for each layer"""
        lstm_layers = []
        input_size = self.config.model_settings.embedding_dim

        for i, hidden_size in enumerate(self.config.model_settings.hidden_dims):
            lstm_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=self.config.model_settings.bidirectional,
                dropout=0
            )
            lstm_layers.append(lstm_layer)
            input_size = hidden_size * \
                (2 if self.config.model_settings.bidirectional else 1)

        return nn.ModuleList(lstm_layers)

    def _create_fc_layers(self) -> nn.Module:
        """Create fully connected layers"""
        last_hidden_size = self.config.model_settings.hidden_dims[-1] * (
            2 if self.config.model_settings.bidirectional else 1)

        return nn.Sequential(
            nn.Linear(last_hidden_size, self.config.model_settings.output_dim),
            self.final_activation if self.final_activation is not None else nn.Identity()
        )

    def apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to the input tensor"""
        if self.attention is None:
            return x

        if isinstance(self.attention, nn.MultiheadAttention):
            x, _ = self.attention(x, x, x)
        else:
            # For other attention types
            if self.config.model_settings.attention_type == 'dot':
                scores = torch.bmm(x, self.attention(x).transpose(1, 2))
                attention = F.softmax(scores, dim=-1)
                x = torch.bmm(attention, x)
            elif self.config.model_settings.attention_type in ['general', 'concat']:
                attention_weights = self.attention(x)
                if self.config.model_settings.attention_type == 'concat':
                    attention_weights = attention_weights.squeeze(-1)
                attention_weights = F.softmax(attention_weights, dim=1)
                x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
            else:  # scaled_dot
                x = self.attention(x)

        return x

    def init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for each LSTM layer"""
        hidden_states = []
        for hidden_size in self.config.model_settings.hidden_dims:
            num_directions = 2 if self.config.model_settings.bidirectional else 1
            weight = next(self.parameters())
            h0 = weight.new_zeros(num_directions, batch_size, hidden_size)
            c0 = weight.new_zeros(num_directions, batch_size, hidden_size)
            hidden_states.append((h0, c0))
        return hidden_states

    def _create_res_projections(self) -> Optional[nn.ModuleList]:
        """Create projection layers for residual connections when dimensions don't match"""
        if not self.config.model_settings.use_res_net:
            return None

        projections = []
        input_size = self.config.model_settings.embedding_dim
        for hidden_size in self.config.model_settings.hidden_dims:
            out_size = hidden_size * \
                (2 if self.config.model_settings.bidirectional else 1)
            if input_size != out_size:
                projections.append(nn.Linear(input_size, out_size))
            else:
                projections.append(nn.Identity())
            input_size = out_size

        return nn.ModuleList(projections)

    @abstractmethod
    def forward(self, x: torch.Tensor,
                hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[
            torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Abstract forward method to be implemented by subclasses"""
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = super().get_model_info()
        # Add residual network information
        if self.config.model_settings.use_res_net:
            info.update({
                'use_res_net': True,
                'res_dropout': self.config.model_settings.res_dropout
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
