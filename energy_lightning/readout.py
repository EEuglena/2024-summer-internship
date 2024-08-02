import torch
import torch.nn as nn


class ReadoutLayer(nn.Module):
    def __init__(
        self,
        c_dim,
        readout_dim,
        # std,
        # mu,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.readout_dim = readout_dim
        # self.std = std
        # self.mu = mu
        self.co = nn.Linear(self.c_dim, self.readout_dim)
        self.o1 = nn.Linear(self.readout_dim, 1)
        # torch.nn.init.normal_(self.o1.weight.data, mean=mu, std=std)

    def forward(self, batch):
        o = torch.tanh(self.co(batch["x"]))
        E = self.o1(o)
        E = torch.sum(E, 1)
        return E
