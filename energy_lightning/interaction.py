import torch
import torch.nn as nn


class InteractionLayer(nn.Module):
    def __init__(self, c_dim, d_dim, interaction_dim):
        super().__init__()
        self.c_dim = c_dim
        self.d_dim = d_dim
        self.interaction_dim = interaction_dim

        self.cf = nn.Linear(self.c_dim, self.interaction_dim)
        self.df = nn.Linear(self.d_dim, self.interaction_dim)
        self.fc = nn.Linear(self.interaction_dim, self.c_dim, bias=False)

    def forward(self, batch):
        cf = self.cf(batch["x"])
        df = self.df(batch["r"])
        fc = self.fc(torch.einsum("...ik,...ijk->...ik", cf, df))
        v = torch.tanh(fc)
        batch["x"] = batch["x"] + v
        return batch
