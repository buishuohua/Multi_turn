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
            self.hidden_dim = config.model_settings.custom_hidden_dims[-1]

        # Multi-head attention parameters
        self.num_heads = config.model_settings.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = math.sqrt(self.head_dim)
        self.temperature = config.model_settings.attention_temperature

        # Create trainable linear transformations for each head
        self.q_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.head_dim,
                      bias=True).to(self.device)
            for _ in range(self.num_heads)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.head_dim,
                      bias=True).to(self.device)
            for _ in range(self.num_heads)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.head_dim,
                      bias=True).to(self.device)
            for _ in range(self.num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim).to(self.device)

        # Initialize parameters with Xavier/Glorot initialization
        for head_idx in range(self.num_heads):
            nn.init.xavier_uniform_(self.q_projs[head_idx].weight)
            nn.init.xavier_uniform_(self.k_projs[head_idx].weight)
            nn.init.xavier_uniform_(self.v_projs[head_idx].weight)
            nn.init.zeros_(self.q_projs[head_idx].bias)
            nn.init.zeros_(self.k_projs[head_idx].bias)
            nn.init.zeros_(self.v_projs[head_idx].bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Dropout and layer norm
        self.dropout = nn.Dropout(
            config.model_settings.attention_dropout).to(self.device)
        self.layer_norm = None  # Will be initialized in forward pass

        # Dynamic projections
        self.input_projection = None
        self.output_projection = None

    def forward(self, x):
        x = x.to(self.device)
        original_x = x  # Store for residual
        batch_size, seq_length, input_dim = x.size()

        # Initialize layer norm if needed
        if self.layer_norm is None or self.layer_norm.normalized_shape[0] != input_dim:
            self.layer_norm = nn.LayerNorm(input_dim).to(self.device)

        # Handle dimension mismatch
        if input_dim != self.hidden_dim:
            if self.input_projection is None or self.input_projection.in_features != input_dim:
                self.input_projection = nn.Linear(
                    input_dim, self.hidden_dim).to(self.device)
            x = self.input_projection(x)

        # Apply layer normalization
        x_norm = self.layer_norm(original_x)
        if input_dim != self.hidden_dim:
            x_norm = self.input_projection(x_norm)

        # Process each head separately
        head_outputs = []
        for head_idx in range(self.num_heads):
            # Project input for this head
            # [batch_size, seq_length, head_dim]
            q = self.q_projs[head_idx](x_norm)
            # [batch_size, seq_length, head_dim]
            k = self.k_projs[head_idx](x_norm)
            # [batch_size, seq_length, head_dim]
            v = self.v_projs[head_idx](x_norm)

            # Compute attention scores for this head
            scores = torch.matmul(q, k.transpose(-2, -1)) / \
                (self.scale * self.temperature)
            attn_weights = self.dropout(F.softmax(scores, dim=-1))
            # [batch_size, seq_length, head_dim]
            head_output = torch.matmul(attn_weights, v)
            head_outputs.append(head_output)

        # Concatenate and pool head outputs
        # [batch_size, seq_length, hidden_dim]
        multi_head_output = torch.cat(head_outputs, dim=-1)
        output = self.out_proj(multi_head_output)

        # Project back to input dimension if needed
        if input_dim != self.hidden_dim:
            if self.output_projection is None or self.output_projection.out_features != input_dim:
                self.output_projection = nn.Linear(
                    self.hidden_dim, input_dim).to(self.device)
            output = self.output_projection(output)

        # Residual connection
        return original_x + output

    def to(self, device):
        """Moves all model parameters and buffers to the specified device"""
        super().to(device)
        self.device = device

        # Move all head-specific projections
        for head_idx in range(self.num_heads):
            self.q_projs[head_idx] = self.q_projs[head_idx].to(device)
            self.k_projs[head_idx] = self.k_projs[head_idx].to(device)
            self.v_projs[head_idx] = self.v_projs[head_idx].to(device)

        # Move other components
        if self.layer_norm is not None:
            self.layer_norm = self.layer_norm.to(device)
        if self.input_projection is not None:
            self.input_projection = self.input_projection.to(device)
        if self.output_projection is not None:
            self.output_projection = self.output_projection.to(device)
        return self
