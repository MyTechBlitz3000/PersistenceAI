# model/config.py

from dataclasses import dataclass

@dataclass
class PersistenceConfig:
    vocab_size: int = 32000
    context_length: int = 2048

    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512

    dropout: float = 0.1

    bias: bool = False
