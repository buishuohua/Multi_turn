# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/11 13:25
    @ Description:
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union, Dict
import torch.nn as nn
import torch
from transformers import AutoConfig


@dataclass
class ModelSettings:
    """Model architecture settings"""
    output_dim: Optional[int] = None  # Make it optional, will be set by Trainer
    embedding_dim: Optional[int] = None  # Now optional
    embedding_type: Literal['BERT_base_uncased', 'BERT_large_uncased',
                            'RoBERTa_base', 'glove_100d', 'glove_300d'] = 'BERT_base_uncased'

    # Optional parameters (with defaults)
    init_hidden_dim: Optional[int] = 256  # Make it optional
    custom_hidden_dims: Optional[List[int]] = None
    # This will store the actual hidden dims used
    hidden_dims: List[int] = field(init=False)
    dropout_rate: float = 0.1
    num_layers: int = 2
    bidirectional: bool = True

    # Fine-tuning settings with better defaults for BERT
    fine_tune_embedding: bool = False
    fine_tune_mode: Literal['none', 'full',
                            'last_n', 'gradual', 'selective'] = 'gradual'
    fine_tune_lr: float = 5e-5  # Standard BERT fine-tuning learning rate

    # Mode-specific parameters
    # For 'last_n' mode - BERT has 12 layers
    # Freeze first 8 layers, fine-tune last 4
    num_frozen_layers: Optional[int] = 8

    # For 'gradual' mode
    gradual_unfreeze_epochs: Optional[int] = 100  # Unfreeze every 100 epochs
    gradual_lr_multiplier: float = 0.95  # Slightly reduced from 0.9

    # For 'selective' mode - targeting specific BERT components
    selective_layers: Optional[List[int]] = field(
        default_factory=lambda: [9, 10, 11]  # Default layers to fine-tune
    )
    # Will be set in post_init
    layer_lr_multipliers: Optional[Dict[int, float]] = None

    # Advanced fine-tuning options
    use_discriminative_lr: bool = True  # Enable by default for BERT
    lr_decay_factor: float = 0.95  # Gradual learning rate decay
    warmup_steps: int = 1000  # More warmup steps for BERT
    gradient_scale: float = 2.0  # Increased from 1.0 to help with vanishing gradients
    max_grad_norm: float = 1.0  # Keep standard clipping

    activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = 'relu'
    final_activation: Literal['softmax', 'sigmoid', 'linear'] = 'softmax'
    loss: Literal['cross_entropy',              # Standard cross-entropy
                  # BCE with logits (recommended for multi-label)
                  'bce_with_logits',
                  'weighted_cross_entropy',      # For class imbalance
                  'focal',                       # For highly imbalanced datasets
                  'label_smoothing_ce',         # Cross-entropy with label smoothing
                  'kl_div',                     # KL divergence loss
                  'mse',                        # Mean squared error
                  'mae'] = 'cross_entropy'      # Mean absolute error
    pooling: Literal['last', 'mean'] = 'last'
    weight_init: Literal['xavier_uniform', 'xavier_normal',
                         'kaiming_uniform', 'kaiming_normal',
                         'orthogonal', 'normal', 'uniform'] = 'orthogonal'
    init_gain: float = 1.0  # Gain parameter for xavier/orthogonal init
    init_mean: float = 0.0  # For normal initialization
    init_std: float = 0.02  # For normal initialization
    init_a: float = -0.1   # For uniform initialization lower bound
    init_b: float = 0.1    # For uniform initialization upper bound

    # Add new loss-related parameters
    focal_alpha: float = 1.0                    # Focal loss alpha parameter
    focal_gamma: float = 2.0                    # Focal loss gamma parameter
    label_smoothing: float = 0.1               # Label smoothing factor
    # Class weights for weighted losses
    class_weights: Optional[List[float]] = None

    # Add new attention-related parameters
    use_attention: bool = True
    attention_type: Literal['dot', 'general', 'concat',
                            'scaled_dot', 'multi_head'] = 'scaled_dot'
    attention_dim: Optional[int] = None  # Will default to hidden_dim if None
    num_attention_heads: int = 8  # For multi-head attention
    attention_dropout: float = 0.1
    attention_temperature: float = 1.0  # Scaling factor for attention scores

    # Add residual network parameter
    use_res_net: bool = False  # Whether to use residual connections between LSTM layers
    res_dropout: float = 0.1   # Dropout rate for residual connections

    # Add layer normalization settings
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    # True for elementwise, False for layerwise
    layer_norm_elementwise: bool = False
    layer_norm_affine: bool = True  # Whether to use learnable affine parameters

    def __post_init__(self):
        """Validate settings after initialization"""
        # First validate and set embedding dimension
        self._validate_and_set_embedding_dim()

        # Then validate other settings
        self._validate_settings()

        # Original post_init logic
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        # Modified hidden dimensions logic
        if self.custom_hidden_dims is not None:
            if len(self.custom_hidden_dims) != self.num_layers:
                raise ValueError(f"Number of custom hidden dimensions ({len(self.custom_hidden_dims)}) "
                                 f"must match number of layers ({self.num_layers})")
            if not all(isinstance(dim, int) and dim > 0 for dim in self.custom_hidden_dims):
                raise ValueError(
                    "All custom hidden dimensions must be positive integers")
            self.hidden_dims = self.custom_hidden_dims
            self.init_hidden_dim = None  # Set to None when using custom dims
        else:
            if self.init_hidden_dim is None:
                raise ValueError(
                    "Either init_hidden_dim or custom_hidden_dims must be provided")
            self.hidden_dims = [self.init_hidden_dim //
                                (2**i) for i in range(self.num_layers)]

        if not all(isinstance(dim, int) and dim > 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive integers")

        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError("output_dim must be positive")

        # Set attention dimension if not provided
        if self.use_attention and self.attention_dim is None:
            self.attention_dim = self.init_hidden_dim

        # Validate attention settings
        self._validate_attention_settings()

        # Validate fine-tuning settings
        if self.fine_tune_embedding:
            self._validate_fine_tune_settings()

        # Dynamically set layer_lr_multipliers based on selective_layers
        if self.fine_tune_mode == 'selective' and self.selective_layers:
            if self.layer_lr_multipliers is None:
                # Create multipliers with gradually decreasing values
                num_layers = len(self.selective_layers)
                self.layer_lr_multipliers = {
                    # Decrease by 5% for each earlier layer
                    layer: 1.0 - (0.05 * i)
                    for i, layer in enumerate(sorted(self.selective_layers, reverse=True))
                }

    def _get_model_name(self) -> str:
        """Convert embedding type to huggingface model name"""
        model_names = {
            'BERT_base_uncased': 'bert-base-uncased',
            'BERT_large_uncased': 'bert-large-uncased',
            'RoBERTa_base': 'roberta-base',
            'RoBERTa_large': 'roberta-large',
        }
        if self.embedding_type not in model_names:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        return model_names[self.embedding_type]

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension based on model type"""
        try:
            # Handle transformer models
            if any(model_type in self.embedding_type
                   for model_type in ['BERT', 'RoBERTa']):
                config = AutoConfig.from_pretrained(self._get_model_name())
                return config.hidden_size

            # Handle GloVe embeddings
            elif 'glove' in self.embedding_type.lower():
                try:
                    return int(self.embedding_type.lower().split('_')[1].replace('d', ''))
                except (IndexError, ValueError):
                    raise ValueError(
                        f"Invalid GloVe embedding format. Expected format: 'glove_<dim>d'"
                    )
            else:
                raise ValueError(
                    f"Unknown embedding type: {self.embedding_type}")

        except Exception as e:
            raise ValueError(f"Error getting embedding dimension: {str(e)}")

    def _validate_and_set_embedding_dim(self):
        """Validate and set embedding dimension"""
        try:
            # Get the model's actual embedding dimension
            model_embedding_dim = self._get_embedding_dim()

            # If embedding_dim is not provided, use the model's dimension
            if self.embedding_dim is None:
                self.embedding_dim = model_embedding_dim
            # If embedding_dim is provided, validate it matches the model
            elif self.embedding_dim != model_embedding_dim:
                raise ValueError(
                    f"Provided embedding_dim ({self.embedding_dim}) does not match "
                    f"{self.embedding_type}'s dimension ({model_embedding_dim})"
                )
        except Exception as e:
            raise ValueError(f"Error validating embedding dimension: {str(e)}")

    def _validate_settings(self):
        """Validate all model settings"""
        errors = []

        # Validate dropout rate
        if not 0 <= self.dropout_rate <= 1:
            errors.append("dropout_rate must be between 0 and 1")

        # Validate dimensions
        if self.output_dim is not None and self.output_dim <= 0:
            errors.append("output_dim must be positive")

        if self.init_hidden_dim <= 0:
            errors.append("init_hidden_dim must be positive")

        # Validate number of layers
        if self.num_layers <= 0:
            errors.append("num_layers must be positive")

        # Validate initialization parameters
        if self.init_std <= 0:
            errors.append("init_std must be positive")

        if self.init_b <= self.init_a:
            errors.append("init_b must be greater than init_a")

        # If there are any errors, raise them all at once
        if errors:
            raise ValueError("\n".join(errors))

    def _validate_fine_tune_settings(self):
        """Validate fine-tuning settings based on selected mode"""
        if self.fine_tune_mode == 'none':
            if self.fine_tune_embedding:
                raise ValueError(
                    "fine_tune_mode 'none' is incompatible with fine_tune_embedding=True")

        elif self.fine_tune_mode == 'last_n':
            if self.num_frozen_layers is None:
                raise ValueError(
                    "num_frozen_layers must be specified for 'last_n' mode")
            if self.num_frozen_layers < 0:
                raise ValueError("num_frozen_layers must be non-negative")

        elif self.fine_tune_mode == 'gradual':
            if self.gradual_unfreeze_epochs is None:
                raise ValueError(
                    "gradual_unfreeze_epochs must be specified for 'gradual' mode")
            if self.gradual_unfreeze_epochs <= 0:
                raise ValueError("gradual_unfreeze_epochs must be positive")
            if not 0 < self.gradual_lr_multiplier <= 1:
                raise ValueError(
                    "gradual_lr_multiplier must be between 0 and 1")

        elif self.fine_tune_mode == 'selective':
            if self.selective_layers is None:
                raise ValueError(
                    "selective_layers must be specified for 'selective' mode")
            if not all(isinstance(layer, int) and layer >= 0 for layer in self.selective_layers):
                raise ValueError(
                    "selective_layers must be non-negative integers")

            # Validate layer_lr_multipliers matches selective_layers
            if self.layer_lr_multipliers:
                if set(self.layer_lr_multipliers.keys()) != set(self.selective_layers):
                    raise ValueError(
                        "layer_lr_multipliers keys must match selective_layers exactly")
                if not all(0 < mult <= 1 for mult in self.layer_lr_multipliers.values()):
                    raise ValueError(
                        "All learning rate multipliers must be between 0 and 1")

        # Validate common parameters
        if self.fine_tune_lr <= 0:
            raise ValueError("fine_tune_lr must be positive")
        if self.gradient_scale <= 0:
            raise ValueError("gradient_scale must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    def get_activation(self) -> nn.Module:
        """Return the appropriate activation function"""

        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations[self.activation]

    def get_final_activation(self) -> Optional[nn.Module]:
        """Return the final activation function"""
        if self.final_activation == 'none':
            return None
        activations = {
            'softmax': nn.Softmax(dim=1),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(self.final_activation)

    def get_loss(self) -> nn.Module:
        """Return the appropriate loss function"""
        if self.loss == 'cross_entropy':
            return nn.CrossEntropyLoss(
                weight=torch.tensor(
                    self.class_weights) if self.class_weights else None
            )

        elif self.loss == 'bce_with_logits':
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    self.class_weights) if self.class_weights else None
            )

        elif self.loss == 'weighted_cross_entropy':
            if not self.class_weights:
                raise ValueError(
                    "class_weights must be provided for weighted_cross_entropy")
            return nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights))

        elif self.loss == 'focal':
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1, gamma=2):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma

                def forward(self, inputs, targets):
                    ce_loss = nn.CrossEntropyLoss()(inputs, targets)
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss

            return FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)

        elif self.loss == 'label_smoothing_ce':
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        elif self.loss == 'kl_div':
            return nn.KLDivLoss(reduction='batchmean')

        elif self.loss == 'mse':
            return nn.MSELoss()

        elif self.loss == 'mae':
            return nn.L1Loss()

        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def get_initializer(self) -> Dict:
        """Return initialization configuration"""
        init_config = {
            'method': self.weight_init,
            'params': {
                'gain': self.init_gain,
                'mean': self.init_mean,
                'std': self.init_std,
                'a': self.init_a,
                'b': self.init_b
            }
        }
        return init_config

    def _validate_attention_settings(self):
        """Validate attention-related settings"""
        if self.use_attention:
            errors = []

            if self.attention_type == 'multi_head':
                if self.num_attention_heads <= 0:
                    errors.append("num_attention_heads must be positive")
                if self.attention_dim % self.num_attention_heads != 0:
                    errors.append(
                        "attention_dim must be divisible by num_attention_heads")

            if self.attention_dim <= 0:
                errors.append("attention_dim must be positive")

            if not 0 <= self.attention_dropout <= 1:
                errors.append("attention_dropout must be between 0 and 1")

            if self.attention_temperature <= 0:
                errors.append("attention_temperature must be positive")

            if errors:
                raise ValueError("\n".join(errors))

    def get_attention_config(self) -> Dict:
        """Return attention configuration"""
        if not self.use_attention:
            return {'use_attention': False}

        return {
            'use_attention': True,
            'type': self.attention_type,
            'dim': self.attention_dim,
            'num_heads': self.num_attention_heads,
            'dropout': self.attention_dropout,
            'temperature': self.attention_temperature
        }

    def get_fine_tuning_config(self) -> Dict:
        """Get fine-tuning configuration based on selected mode"""
        if not self.fine_tune_embedding:
            return {'enabled': False}

        config = {
            'enabled': True,
            'mode': self.fine_tune_mode,
            'base_lr': self.fine_tune_lr,
            'gradient_scale': self.gradient_scale,
            'max_grad_norm': self.max_grad_norm,
            'warmup_steps': self.warmup_steps
        }

        # Add mode-specific configurations
        if self.fine_tune_mode == 'last_n':
            config.update({
                'num_frozen_layers': self.num_frozen_layers
            })

        elif self.fine_tune_mode == 'gradual':
            config.update({
                'unfreeze_epochs': self.gradual_unfreeze_epochs,
                'lr_multiplier': self.gradual_lr_multiplier
            })

        elif self.fine_tune_mode == 'selective':
            config.update({
                'selective_layers': self.selective_layers,
                'layer_lr_multipliers': self.layer_lr_multipliers
            })

        if self.use_discriminative_lr:
            config.update({
                'discriminative_lr': True,
                'lr_decay_factor': self.lr_decay_factor
            })

        return config

    @classmethod
    def get_default(cls):
        return cls(
            output_dim=None,  # Will be set automatically by Trainer
            embedding_dim=None,  # Will be set based on embedding type
            embedding_type='BERT_base_uncased',
            init_hidden_dim=256,
            dropout_rate=0.1,
            num_layers=2,
            bidirectional=True,
            fine_tune_embedding=False,
            fine_tune_mode='none',
            fine_tune_lr=2e-5,
            activation='relu',
            final_activation='softmax',
            loss='cross_entropy',
            pooling='last',
            weight_init='orthogonal',
            init_gain=1.0,
            init_mean=0.0,
            init_std=0.02,
            init_a=-0.1,
            init_b=0.1,
            focal_alpha=1.0,
            focal_gamma=2.0,
            label_smoothing=0.1,
            class_weights=None,
            # Default attention settings
            use_attention=True,
            attention_type='scaled_dot',
            attention_dim=None,
            num_attention_heads=8,
            attention_dropout=0.1,
            attention_temperature=1.0,
            # Default residual network settings
            use_res_net=False,
            res_dropout=0.1
        )

    @classmethod
    def get_default_fine_tuning(cls) -> 'ModelSettings':
        """Get optimized default fine-tuning settings for BERT"""
        return cls(
            fine_tune_embedding=True,
            fine_tune_mode='gradual',  # Gradual unfreezing works well with BERT
            fine_tune_lr=2e-5,

            # Gradual unfreezing settings
            gradual_unfreeze_epochs=2,
            gradual_lr_multiplier=0.95,

            # Layer-specific settings
            num_frozen_layers=8,  # Start with last 4 layers unfrozen
            selective_layers=[9, 10, 11],  # Focus on last 3 layers
            layer_lr_multipliers={
                11: 1.0,
                10: 0.95,
                9: 0.9,
            },

            # Advanced settings
            use_discriminative_lr=True,
            lr_decay_factor=0.95,
            warmup_steps=1000,
            gradient_scale=2.0,
            max_grad_norm=1.0,

            # Additional settings for stability
            use_layer_norm=True,
            layer_norm_eps=1e-5,
            use_attention=True,
            attention_dropout=0.1,
            dropout_rate=0.1
        )

    def to_dict(self) -> dict:
        """Convert ModelSettings to a dictionary"""
        return {
            'output_dim': self.output_dim,
            'embedding_dim': self.embedding_dim,
            'embedding_type': self.embedding_type,
            'init_hidden_dim': self.init_hidden_dim,
            'custom_hidden_dims': self.custom_hidden_dims,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'fine_tune_embedding': self.fine_tune_embedding,
            'fine_tune_mode': self.fine_tune_mode,
            'fine_tune_lr': self.fine_tune_lr,
            'activation': self.activation,
            'final_activation': self.final_activation,
            'loss': self.loss,
            'pooling': self.pooling,
            'weight_init': self.weight_init,
            'init_gain': self.init_gain,
            'init_mean': self.init_mean,
            'init_std': self.init_std,
            'init_a': self.init_a,
            'init_b': self.init_b,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            'label_smoothing': self.label_smoothing,
            'class_weights': self.class_weights,
            # Attention settings
            'use_attention': self.use_attention,
            'attention_type': self.attention_type,
            'attention_dim': self.attention_dim,
            'num_attention_heads': self.num_attention_heads,
            'attention_dropout': self.attention_dropout,
            'attention_temperature': self.attention_temperature,
            # Residual network settings
            'use_res_net': self.use_res_net,
            'res_dropout': self.res_dropout,
            # Layer normalization settings
            'use_layer_norm': self.use_layer_norm,
            'layer_norm_eps': self.layer_norm_eps,
            'layer_norm_elementwise': self.layer_norm_elementwise,
            'layer_norm_affine': self.layer_norm_affine
        }
