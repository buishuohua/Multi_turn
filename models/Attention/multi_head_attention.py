import torch
import torch.nn as nn
import math
from transformers import AutoModel
from typing import Optional, Tuple
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.training_settings.device

        # Use BERT's hidden size as input dimension
        self.hidden_dim = config.model_settings.embedding_dim
        if self.hidden_dim is None:
            # If embedding_dim is not set, use the last dimension from custom_hidden_dims
            self.hidden_dim = config.model_settings.custom_hidden_dims[-1]

        # Multi-head attention parameters
        self.num_heads = config.model_settings.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = math.sqrt(self.head_dim)

        # Add temperature parameter
        self.temperature = config.model_settings.attention_temperature

        # Linear projections
        self.q_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim).to(self.device)
        self.k_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim).to(self.device)
        self.v_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim).to(self.device)
        self.out_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim).to(self.device)

        # Dropout and layer norm
        self.dropout = nn.Dropout(
            config.model_settings.attention_dropout).to(self.device)
        # Layer norm should match the input feature dimension
        self.layer_norm = None  # Will be initialized in forward pass

        # Input/output projections will be created dynamically
        self.input_projection = None
        self.output_projection = None

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        original_x = x  # Store original input for residual connection

        batch_size, seq_length, input_dim = x.size()

        # Initialize layer norm if needed
        if self.layer_norm is None or self.layer_norm.normalized_shape[0] != input_dim:
            self.layer_norm = nn.LayerNorm(input_dim).to(self.device)

        # Create or update input projection if needed
        if input_dim != self.hidden_dim:
            if self.input_projection is None or self.input_projection.in_features != input_dim:
                self.input_projection = nn.Linear(
                    input_dim, self.hidden_dim).to(self.device)
            x = self.input_projection(x)

        # Apply layer normalization first (pre-norm)
        x_norm = self.layer_norm(original_x)  # Apply norm to original input

        # Project normalized input if dimensions don't match
        if input_dim != self.hidden_dim:
            x_norm = self.input_projection(x_norm)

        # Linear projections and reshape for multi-head
        q = self.q_proj(x_norm).view(batch_size, seq_length,
                                     self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_length,
                                     self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_length,
                                     self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            (self.scale * self.temperature)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_dim)
        output = self.out_proj(context)

        # Create or update output projection if needed
        if input_dim != self.hidden_dim:
            if self.output_projection is None or self.output_projection.out_features != input_dim:
                self.output_projection = nn.Linear(
                    self.hidden_dim, input_dim).to(self.device)
            output = self.output_projection(output)

        # Residual connection with original input
        return original_x + output

    def to(self, device):
        """Moves all model parameters and buffers to the specified device"""
        super().to(device)
        self.device = device
        if self.layer_norm is not None:
            self.layer_norm = self.layer_norm.to(device)
        if self.input_projection is not None:
            self.input_projection = self.input_projection.to(device)
        if self.output_projection is not None:
            self.output_projection = self.output_projection.to(device)
        return self
