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
            nn.Linear(self.hidden_dim, self.head_dim, bias=True)
            for _ in range(self.num_heads)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.head_dim, bias=True)
            for _ in range(self.num_heads)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.head_dim, bias=True)
            for _ in range(self.num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Dropout and layer norm
        self.dropout = nn.Dropout(config.model_settings.attention_dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize attention parameters"""
        # Initialize layer norm parameters first
        nn.init.ones_(self.layer_norm.weight)  # Initialize to 1
        nn.init.zeros_(self.layer_norm.bias)   # Initialize to 0

        # Initialize attention head parameters
        for head_idx in range(self.num_heads):
            nn.init.xavier_uniform_(self.q_projs[head_idx].weight)
            nn.init.xavier_uniform_(self.k_projs[head_idx].weight)
            nn.init.xavier_uniform_(self.v_projs[head_idx].weight)
            nn.init.zeros_(self.q_projs[head_idx].bias)
            nn.init.zeros_(self.k_projs[head_idx].bias)
            nn.init.zeros_(self.v_projs[head_idx].bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        original_x = x  # Store for residual
        batch_size, seq_length, input_dim = x.size()
        device = x.device

        # Ensure layer norm is on the correct device
        if self.layer_norm.weight.device != device:
            self.to(device)

        # Apply layer normalization
        x_norm = self.layer_norm(x)

        # Process each head separately
        head_outputs = []
        for head_idx in range(self.num_heads):
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

        # Residual connection
        return original_x + output

    def to(self, device):
        """Moves all model parameters and buffers to the specified device"""
        # Move base module
        super().to(device)

        # Move layer norm
        if self.layer_norm is not None:
            self.layer_norm = self.layer_norm.to(device)

        # Move projections
        for i in range(self.num_heads):
            self.q_projs[i] = self.q_projs[i].to(device)
            self.k_projs[i] = self.k_projs[i].to(device)
            self.v_projs[i] = self.v_projs[i].to(device)

        # Move output projection
        if self.out_proj is not None:
            self.out_proj = self.out_proj.to(device)

        return self
