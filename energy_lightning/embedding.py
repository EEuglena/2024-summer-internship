import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_dim, c_dim, d_dim):
        super().__init__()
        self.embed_dim = c_dim
        self.c_dim = c_dim
        self.d_dim = d_dim
        # self.zc = nn.Linear(self.embed_dim, self.c_dim)
        self.rd = nn.Linear(1, self.d_dim)

    def forward(self, batch):
        batch = self._one_hot_encoding(batch)
        # batch = self._atomic_embedding(batch)
        batch = self._distance_embedding(batch)
        return batch

    def _one_hot_encoding(self, batch):
        batch["x"] = one_hot(batch["z"], self.embed_dim).float().requires_grad_(False)
        return batch

    def _atomic_embedding(self, batch):
        batch["x"] = self.zc(batch["x"])
        return batch

    def _distance_embedding(self, batch):
        batch["r"] = self.rd(batch["r"].unsqueeze(-1))
        return batch
