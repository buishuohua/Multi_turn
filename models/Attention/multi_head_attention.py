import torch
import torch.nn as nn
import math
from transformers import AutoModel
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.model_settings.hidden_dims[-1] * (
            2 if config.model_settings.bidirectional else 1)

        # Get pre-trained model based on embedding type
        model_name = self._get_pretrained_model_name()
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        # Get attention from the same model as embeddings
        attention = self.pretrained_model.encoder.layer[-1].attention.self

        # Extract Q,K,V matrices
        self.query = attention.query
        self.key = attention.key
        self.value = attention.value

        # Freeze if not fine-tuning
        if not config.model_settings.fine_tune_embedding:
            for param in self.parameters():
                param.requires_grad = False

        self.num_heads = config.model_settings.num_attention_heads
        self.dropout = nn.Dropout(config.model_settings.attention_dropout)
        self.scale = math.sqrt(self.hidden_dim // self.num_heads)

    def _get_pretrained_model_name(self):
        """Convert embedding type to model name"""
        mapping = {
            'BERT_base_uncased': 'bert-base-uncased',
            'BERT_large_uncased': 'bert-large-uncased',
            'RoBERTa_base': 'roberta-base',
            'RoBERTa_large': 'roberta-large'
        }
        model_type = self.config.model_settings.embedding_type
        if model_type not in mapping:
            raise ValueError(
                f"Unsupported embedding type for attention: {model_type}")
        return mapping[model_type]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = x.size()

        # Project input using pre-trained Q,K,V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply attention dropout
        attention_weights = self.dropout(torch.softmax(scores, dim=-1))

        # Get attention output
        context = torch.matmul(attention_weights, V)

        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous()
        output = context.view(batch_size, seq_length, self.hidden_dim)

        return output, attention_weights
