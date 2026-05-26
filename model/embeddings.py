# model/embeddings.py

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, n_embd: int):
        super().__init__()

        self.embedding = nn.Embedding(context_length, n_embd)

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(0, T, device=x.device)
        positions = positions.unsqueeze(0)

        return self.embedding(positions)
