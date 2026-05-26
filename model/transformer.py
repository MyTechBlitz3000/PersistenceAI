# model/transformer.py

import torch
import torch.nn as nn

from model.config import PersistenceConfig
from model.embeddings import TokenEmbedding, PositionalEmbedding
from model.attention import CausalSelfAttention
from model.mlp import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config: PersistenceConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            config.n_embd,
            config.n_head,
            config.context_length,
            config.dropout
        )

        self.ln2 = nn.LayerNorm(config.n_embd)

        self.ff = FeedForward(
            config.n_embd,
            config.dropout
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x


class PersistenceAI(nn.Module):
    def __init__(self, config: PersistenceConfig):
        super().__init__()

        self.config = config

        self.token_embedding = TokenEmbedding(
            config.vocab_size,
            config.n_embd
        )

        self.position_embedding = PositionalEmbedding(
            config.context_length,
            config.n_embd
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(idx)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits
