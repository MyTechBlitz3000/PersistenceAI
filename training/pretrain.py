import torch
import torch.nn.functional as F
import sentencepiece as spm

from torch.utils.data import DataLoader

from model.config import PersistenceConfig
from model.transformer import PersistenceAI
from training.dataset import TextDataset

config = PersistenceConfig()

device = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

with open("datasets/data.txt", "r") as f:
    text = f.read()

tokens = sp.encode(text)

dataset = TextDataset(
    tokens,
    context_length=128
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = PersistenceAI(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

for epoch in range(5):

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")
