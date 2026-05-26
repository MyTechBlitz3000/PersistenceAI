# inference/generate.py

import torch

from model.config import PersistenceConfig
from model.transformer import PersistenceAI

config = PersistenceConfig()

model = PersistenceAI(config)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

start_tokens = torch.randint(
    0,
    config.vocab_size,
    (1, 16)
).to(device)

with torch.no_grad():
    logits = model(start_tokens)

print("Output shape:", logits.shape)
