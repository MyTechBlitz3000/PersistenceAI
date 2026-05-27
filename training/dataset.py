import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_length]
        y = self.tokens[idx+1:idx+self.context_length+1]

        return (
            torch.tensor(x),
            torch.tensor(y)
        )
