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
import numpy as np
import re
from models.Attention.multi_head_attention import MultiHeadAttention
from utils.experiment_utils import create_experiment_name
if TYPE_CHECKING:
    from config.Experiment_Config import ExperimentConfig

# Add this import for runtime
from typing import Any as ExperimentConfig  # temporary type alias


class ModelStateManager:
    def __init__(self, config):
        self.config = config
        self.best_state = None
        self.best_val_loss = float('inf')
        self.plateau_counter = 0
        self.improvement_threshold = 1e-4

    def should_reload_model(self, current_val_loss, epoch):
        """Determine if model should be reloaded based on various criteria"""
        # Check for significant improvement
        if current_val_loss < self.best_val_loss - self.improvement_threshold:
            self.best_val_loss = current_val_loss
            self.plateau_counter = 0
            return False

        self.plateau_counter += 1

        # Reload conditions
        if (self.plateau_counter >= 5 or  # Plateau detected
                epoch % max(10, epoch // 5) == 0):  # Adaptive frequency
            self.plateau_counter = 0
            return True

        return False


class CheckpointEnsemble:
    """Manages an ensemble of model checkpoints for improved performance"""

    def __init__(self, max_checkpoints: int = 3, min_improvement: float = 0.01):
        """
        Initialize checkpoint ensemble.

        Args:
            max_checkpoints (int): Maximum number of checkpoints to keep in ensemble
            min_improvement (float): Minimum validation loss improvement required
        """
        self.max_checkpoints = max_checkpoints
        self.min_improvement = min_improvement
        self.checkpoints = []  # List of (state_dict, val_loss) tuples

    def add_checkpoint(self, state_dict: dict, val_loss: float) -> bool:
        """
        Add a new checkpoint to the ensemble if it improves performance.

        Args:
            state_dict: Model state dictionary
            val_loss: Validation loss for this checkpoint

        Returns:
            bool: True if checkpoint was added, False otherwise
        """
        # Always add if we have fewer than max checkpoints
        if len(self.checkpoints) < self.max_checkpoints:
            self.checkpoints.append((state_dict, val_loss))
            # Sort by validation loss
            self.checkpoints.sort(key=lambda x: x[1])
            return True

        # Otherwise, only add if it improves on worst checkpoint by min_improvement
        worst_loss = self.checkpoints[-1][1]
        if val_loss < (worst_loss - self.min_improvement):
            self.checkpoints.pop()  # Remove worst checkpoint
            self.checkpoints.append((state_dict, val_loss))
            self.checkpoints.sort(key=lambda x: x[1])
            return True

        return False

    def get_ensemble_weights(self) -> Optional[dict]:
        """
        Get averaged weights from all checkpoints in ensemble.

        Returns:
            dict: Averaged state dictionary, or None if ensemble is empty
        """
        if not self.checkpoints:
            return None

        # Start with first checkpoint's state dict
        ensemble_state = self.checkpoints[0][0].copy()

        # Add remaining checkpoints
        for state_dict, _ in self.checkpoints[1:]:
            for key in ensemble_state:
                ensemble_state[key] += state_dict[key]

        # Average the parameters
        n = len(self.checkpoints)
        for key in ensemble_state:
            ensemble_state[key] /= n

        return ensemble_state


class BaseLSTM(nn.Module, ABC):
    """
    Base class for LSTM-based architectures.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self.state_manager = ModelStateManager(config)

        # Initialize checkpoint ensemble if ensemble strategy is enabled
        if 'ensemble' in config.model_settings.fine_tune_loading_strategies:
            self.checkpoint_ensemble = CheckpointEnsemble(
                max_checkpoints=config.model_settings.ensemble_max_checkpoints,
                min_improvement=config.model_settings.ensemble_min_improvement
            )
        else:
            self.checkpoint_ensemble = None

        # Create model-specific experiment name
        self.experiment_name = create_experiment_name(config, is_model=False)

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

        # Create layer normalization layers if enabled
        self.layer_norms = self._create_layer_norms(
        ) if config.model_settings.use_layer_norm else None

        # Add attention layer if enabled
        self.attention = None
        if config.model_settings.use_attention:
            self.attention = self._create_attention()

        self.fc_layers = self._create_fc_layers()

        # Initialize weights after creating layers
        self._initialize_weights()

    def _load_or_create_embedding_model(self):
        """Load embedding model with combined strategies"""
        embedding_model = self.config.tokenizer_settings.get_model().model
        self._configure_fine_tuning(embedding_model)

        if self.config.training_settings.continue_training:
            should_load = False

            # Calculate adaptive reload frequency
            base_freq = self.config.model_settings.fine_tune_reload_freq
            # Decrease by 1 every 100 epochs
            adaptive_freq = max(5, base_freq - (self.current_epoch // 100))

            # Check for periodic loading with adaptive frequency
            if 'periodic' in self.config.model_settings.fine_tune_loading_strategies:
                if (self.current_epoch % adaptive_freq == 0):
                    should_load = True
                    print(
                        f"🔄 Adaptive periodic reload triggered at epoch {self.current_epoch} (freq={adaptive_freq})")

            # Check for plateau detection
            if 'plateau' in self.config.model_settings.fine_tune_loading_strategies:
                if hasattr(self, 'val_loss_history') and len(self.val_loss_history) >= self.config.model_settings.plateau_patience:
                    recent_losses = self.val_loss_history[-self.config.model_settings.plateau_patience:]
                    if max(recent_losses) - min(recent_losses) < self.config.model_settings.plateau_threshold:
                        should_load = True
                        print(
                            f"📊 Plateau detected at epoch {self.current_epoch}")

            if should_load and hasattr(self, 'checkpoint_ensemble') and self.checkpoint_ensemble is not None:
                # Get ensemble weights
                try:
                    ensemble_state = self.checkpoint_ensemble.get_ensemble_weights()
                    if ensemble_state is not None:
                        embedding_model.load_state_dict(ensemble_state)
                        print("✨ Loaded ensemble weights")
                    else:
                        self._load_model_weights(embedding_model, 'best')
                        print("⚠️ No ensemble weights available, loaded best weights")
                except Exception as e:
                    print(
                        f"⚠️ Warning: Failed to load ensemble weights: {str(e)}")

        return embedding_model

    def save_checkpoint(self, val_loss):
        """Save current model state to ensemble"""
        if hasattr(self, 'checkpoint_ensemble'):
            self.checkpoint_ensemble.add_checkpoint(
                self.embedding_model.state_dict(),
                val_loss
            )
            print(
                f"📦 Added checkpoint to ensemble (total: {len(self.checkpoint_ensemble.checkpoints)})")

    def _load_model_weights(self, embedding_model, load_type='best'):
        """Load model weights with improved error handling"""
        model_path = os.path.join(
            self.config.training_settings.fine_tuned_models_dir,
            self.config.training_settings.task_type,
            self.config.data_settings.which,
            self.config.model_settings.embedding_type,
            self.experiment_name,
            load_type,
            f'embedding_model_{load_type}.pt'
        )

        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, weights_only=True)
                embedding_model.load_state_dict(checkpoint['state_dict'])
                print(f"✅ Loaded {load_type} fine-tuned embedding model")
                return True
        except Exception as e:
            print(f"⚠️ Error loading {load_type} model: {str(e)}")
        return False

    def _configure_fine_tuning(self, embedding_model):
        """Configure fine-tuning mode for the embedding model"""
        if not self.config.model_settings.fine_tune_embedding:
            for param in embedding_model.parameters():
                param.requires_grad = False
            print("❄️ All embedding layers frozen (no fine-tuning)")
            return

        # First, freeze all parameters
        for param in embedding_model.parameters():
            param.requires_grad = False

        mode = self.config.model_settings.fine_tune_mode

        if mode == 'full':
            # Unfreeze all parameters
            for param in embedding_model.parameters():
                param.requires_grad = True
            print("🔓 Full fine-tuning enabled (all layers unfrozen)")

        elif mode == 'last_n':
            num_frozen = self.config.model_settings.num_frozen_layers
            # Iterate through named parameters to selectively freeze layers
            for name, param in embedding_model.named_parameters():
                layer_match = re.search(r'layer\.(\d+)\.', name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    # Unfreeze if it's after num_frozen layers
                    param.requires_grad = layer_num >= num_frozen

                    # Apply discriminative learning rates if enabled
                    if param.requires_grad and self.config.model_settings.use_discriminative_lr:
                        layer_position = layer_num - num_frozen + 1
                        param.lr_scale = self.config.model_settings.lr_decay_factor ** (
                            layer_position - 1)
                else:
                    # For non-layer parameters (embeddings, pooler, etc.)
                    param.requires_grad = False
            print(f"🔒 Frozen first {num_frozen} layers, fine-tuning the rest")

        elif mode == 'gradual':
            # Start with all frozen - will be gradually unfrozen during training
            # Initial unfreeze of last layer
            for name, param in embedding_model.named_parameters():
                layer_match = re.search(r'layer\.(\d+)\.', name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    # Last layer (BERT has 12 layers, 0-11)
                    if layer_num == 11:
                        param.requires_grad = True
                        if self.config.model_settings.use_discriminative_lr:
                            param.lr_scale = 1.0
            print("🔄 Gradual fine-tuning initialized (starting with last layer)")

        elif mode == 'selective':
            selective_layers = self.config.model_settings.selective_layers
            for name, param in embedding_model.named_parameters():
                layer_match = re.search(r'layer\.(\d+)\.', name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    if layer_num in selective_layers:
                        param.requires_grad = True
                        if self.config.model_settings.use_discriminative_lr:
                            # Get multiplier from config or use default based on position
                            multiplier = self.config.model_settings.layer_lr_multipliers.get(
                                layer_num,
                                self.config.model_settings.lr_decay_factor ** (
                                    len(selective_layers) -
                                    selective_layers.index(layer_num) - 1
                                )
                            )
                            param.lr_scale = multiplier
            print(
                f"🎯 Selective fine-tuning enabled for layers {selective_layers}")

        # Print summary of trainable parameters
        trainable_params = sum(
            p.numel() for p in embedding_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in embedding_model.parameters())
        print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({trainable_params/total_params:.1%})")

    def _create_attention(self) -> Dict[str, nn.Module]:
        """Create attention modules based on configuration"""
        if not self.config.model_settings.use_attention:
            return {}

        attention_modules = {}
        # Attention after embedding
        attention_modules['embedding'] = MultiHeadAttention(self.config)
        # Attention between LSTM layers
        attention_modules['inter_lstm'] = [MultiHeadAttention(self.config)
                                           for _ in range(len(self.config.model_settings.hidden_dims)-1)]
        # Attention after final LSTM
        attention_modules['output'] = MultiHeadAttention(self.config)

        return attention_modules

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
        activation = self.config.model_settings.get_activation()
        # Create sequential layers without final activation
        layers = [activation,
                  nn.Linear(last_hidden_size, self.config.model_settings.output_dim)]

        # Add final activation if specified
        final_activation = self.config.model_settings.get_final_activation()
        if final_activation is not None:
            layers.append(final_activation)

        return nn.Sequential(*layers)

    def apply_attention(self, x: torch.Tensor, attention_module: nn.Module) -> torch.Tensor:
        """Apply specific attention module to the input tensor"""
        if attention_module is None:
            return x

        return attention_module(x)

    def _create_res_projections(self) -> Optional[nn.ModuleList]:
        """Create projection layers for residual connections when dimensions don't match"""
        if not self.config.model_settings.use_res_net:
            return None

        projections = []
        input_size = self.config.model_settings.embedding_dim

        # Create projections for each LSTM layer
        for hidden_size in self.config.model_settings.hidden_dims:
            output_size = hidden_size * \
                (2 if self.config.model_settings.bidirectional else 1)
            if input_size != output_size:
                projections.append(nn.Linear(input_size, output_size))
            else:
                projections.append(nn.Identity())
            input_size = output_size

        return nn.ModuleList(projections)

    def _create_layer_norms(self) -> Optional[nn.ModuleDict]:
        """Create layer normalization layers for LSTM and FC layers"""
        if not self.config.model_settings.use_layer_norm:
            return None

        layer_norms = nn.ModuleDict()

        # Create layer norms for each LSTM layer
        for i, hidden_size in enumerate(self.config.model_settings.hidden_dims):
            output_size = hidden_size * \
                (2 if self.config.model_settings.bidirectional else 1)
            layer_norms[f'lstm_{i}'] = nn.LayerNorm(output_size)

        # Create layer norm for final output
        last_hidden_size = self.config.model_settings.hidden_dims[-1] * (
            2 if self.config.model_settings.bidirectional else 1)
        layer_norms['final'] = nn.LayerNorm(last_hidden_size)

        return layer_norms

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

    def update_gradual_unfreezing(self, epoch: int):
        """Update which layers are frozen/unfrozen during gradual fine-tuning"""
        if not self.config.model_settings.fine_tune_embedding:
            return

        if self.config.model_settings.fine_tune_mode != 'gradual':
            return

        # Get layers to unfreeze at current epoch
        layers_to_unfreeze = self.config.model_settings.get_layers_to_unfreeze_at_epoch(
            epoch)

        # Track changes for logging
        newly_unfrozen = []
        total_unfrozen = []

        # Update layer freezing status
        for name, param in self.embedding_model.named_parameters():
            layer_num = int(re.search(r'layer\.(\d+)\.',
                            name).group(1)) if 'layer.' in name else -1
            if layer_num in layers_to_unfreeze:
                if not param.requires_grad:
                    newly_unfrozen.append(layer_num)
                param.requires_grad = True
                total_unfrozen.append(layer_num)

                # Apply layer-specific learning rate if using discriminative learning rates
                if self.config.model_settings.use_discriminative_lr:
                    layer_position = len(layers_to_unfreeze) - \
                        layers_to_unfreeze.index(layer_num)
                    param.lr_scale = self.config.model_settings.lr_decay_factor ** (
                        layer_position - 1)
            else:
                param.requires_grad = False

        # Log changes
        if newly_unfrozen:
            print(f"🔓 Epoch {epoch}: Unfroze layers {sorted(newly_unfrozen)}")
        print(f"📊 Currently unfrozen layers: {sorted(total_unfrozen)}")

    def update_epoch(self, epoch: int):
        """Update the current epoch number"""
        self.current_epoch = epoch
