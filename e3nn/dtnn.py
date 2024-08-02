from typing import Dict, Any, List
import torch
import e3nn
import torch.utils
from torch_cluster import radius_graph
from torch_geometric.datasets import MD17


class NuclearEmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        max_z: int,
        dim: int,
    ) -> None:
        super().__init__()
        self.one_hot = torch.nn.functional.one_hot
        self.embedding = torch.nn.Linear(max_z, dim)

    def forward(self, atomic_numbers: List[int]) -> torch.Tensor:

        # 1. One-hot Encoding
        one_hot = self.one_hot(atomic_numbers)

        # 2. Linear Transform
        embedded = self.embedding(one_hot)

        return embedded


class GaussianExpansionLayer(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, distances: torch.Tensor) -> torch.Tensor:

        # https://vscode.dev/github/mir-group/nequip/blob/main/nequip/nn/embedding/_edge.py#L61 : Ref.

        # 1. Gaussian Expansion

        return torch.tensor()


class InteractionLayer(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
    ):

        #

        return


class ReadoutLayer(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
    ):

        # 1. Compute energy from x

        return torch.tensor()


class NN(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        data: Dict[str, Any],
    ) -> torch.tensor:
        readout: torch.tensor
        data = self._preprocess(data)
        return readout

    def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:

        # 1. Build Graph

        # 2. Compute r & r_hat

        # 3. Apply Spherical Harmonics

        return data


if __name__ == "__main__":

    # 1. Download Dataset

    # 2. Init Dataloader

    # 3. Init Model

    # 4. Train Model

    # 5. Load Checkpoint

    # 6. Evaluate Model

    print(type(NN.__init__(None)))

    pass
