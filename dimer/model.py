from typing import Callable, Dict, Union, Any
import torch
import numpy as np
import e3nn
from torch_scatter import scatter


class AtomicEmbedding(torch.nn.Module):
    """Embedding layer for atomic numbers"""

    def __init__(
        self,
        device: Union[str, torch.device],
        max_z: int,
        dim_atoms: int,
    ) -> None:
        """

        Args:
            device (Union[str, torch.device]): torch device
            max_z (int): maximum atomic number for one-hot encoding
            dim_atoms (int): dimension of atomic representation
        """
        super().__init__()
        self.device = device
        self.max_z = max_z
        self.one_hot = torch.nn.functional.one_hot
        self.embedding = torch.nn.Linear(max_z, dim_atoms).to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        one_hot = self.one_hot(z, self.max_z).to(self.device).float()
        embed = self.embedding(one_hot)
        return embed


# Copied from DTNN
class GaussianExpansion(torch.nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device],
        mu_min: float,
        mu_max: float,
        step: float,
    ) -> None:
        super().__init__()
        self.device = device
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.step = step

    def _gauss_expand(
        self,
        distance: torch.tensor,
        mu_min: float,
        mu_max: float,
        step: float,
        sigma=None,
    ) -> torch.tensor:  # (n, n) -> (n, n, n_mu)

        mu_range = mu_max - mu_min
        n_mu = int(np.ceil((mu_range / step)) + 1)
        mu = torch.linspace(mu_min, mu_max, n_mu).reshape(1, 1, -1).to(self.device)

        distance = distance.unsqueeze(-1)

        if sigma == None:
            sigma = step
        ged = torch.exp(-0.5 * ((distance - mu) / sigma) ** 2)

        return ged.squeeze(0)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self._gauss_expand(r, self.mu_min, self.mu_max, self.step)


class InteractionLayer(torch.nn.Module):
    """Message passing layer for representing interatomic interations"""

    def __init__(
        self,
        device: Union[str, torch.device],
        irreps_feature: Union[str, e3nn.o3.Irreps],
        irreps_atom: Union[str, e3nn.o3.Irreps],
        activation_fn: Callable,
    ) -> None:
        """

        Args:
            device (Union[str, torch.device]): torch device
            irreps_feature (Union[str, e3nn.o3.Irreps]): irreps for edge feature
            irreps_atom (Union[str, e3nn.o3.Irreps]): irreps for atomic representation
            activation_fn (Callable): activation function
        """
        super().__init__()
        self.device = device
        self.exp = 0.5
        self.s2d = e3nn.o3.FullyConnectedTensorProduct(
            irreps_feature, irreps_atom, irreps_atom
        ).to(device)
        self.d2s = e3nn.o3.FullyConnectedTensorProduct(
            irreps_feature, irreps_atom, irreps_atom
        ).to(device)
        self.activation_fn = activation_fn

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        graph = data["dgraph"]
        N = data["natoms0"] + data["natoms1"]
        s2d = self.s2d(graph["edge_features"], data["y"][0][graph["src"]])
        s2d = s2d / (N**self.exp)
        s2d = self.activation_fn(s2d)
        s2d = scatter(s2d, graph["dst"], dim=0, out=torch.zeros_like(data["y"][1]))

        d2s = self.d2s(graph["edge_features"], data["y"][1][graph["dst"]])
        d2s = d2s / (N**self.exp)
        d2s = self.activation_fn(d2s)
        d2s = scatter(d2s, graph["src"], dim=0, out=torch.zeros_like(data["y"][0]))

        data["y"] = (data["y"][0] + d2s, data["y"][1] + s2d)
        return data


class ReadoutLayer(torch.nn.Module):
    """Readout layer for interpreting atomic representations into molecular potential energy"""

    def __init__(
        self,
        device: Union[str, torch.device],
        dim_atoms,
        activation_fn: Callable,
    ) -> None:
        """

        Args:
            device (Union[str, torch.device]): torch device
            dim_atoms (int): dimension of atomic representation
            dim_mid (int): dimension of intermidiate representation
            activation_fn (Callable): activation function
        """
        super().__init__()
        self.device = device
        self.layer = torch.nn.Linear(dim_atoms, 1)
        self.activation_fn = activation_fn

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = self.layer(torch.cat(data["y"]))
        out = self.activation_fn(out)
        return out.sum()


class DimerInteractionEnergyModel(torch.nn.Module):
    """Neural Network Model that predicts molecular energy from atomic numbers and coordinates"""

    def __init__(
        self,
        device: Union[str, torch.device],
        activation_fn: Callable,
        max_z: int,
        dim_atoms: int,
        mu_min: float,
        mu_max: float,
        step: float,
        irreps_r: Union[str, e3nn.o3.Irreps],
        irreps_sh: Union[str, e3nn.o3.Irreps],
        irreps_atom: Union[str, e3nn.o3.Irreps],
        n_interactions: int,
    ) -> None:
        """

        Args:
            device (Union[str, torch.device]): torch device
            activation_fn (Callable): activation function for model
            r_cut (float): radial cutoff for generating graph
            max_z (int): maximum atomic number for one-hot encoding
            dim_atoms (int): dimension of atomic representation
            dim_mid (int): dimension of intermediate representation
            mu_min (float): lower range of gaussian expansion
            mu_max (float): upper range of gaussian expansion
            step (float): number of bins for gaussian expansion
            irreps_feature (Union[str, e3nn.o3.Irreps]): irreps for edge feature
            irreps_atom (Union[str, e3nn.o3.Irreps]): irreps for atomic representation
            n_interactions (int): number of interacion layers
        """
        super().__init__()
        self.device = device
        self.activation_fn = activation_fn
        self.irreps_r = e3nn.o3.Irreps(irreps_r)
        self.irreps_sh = e3nn.o3.Irreps(irreps_sh)
        self.irreps_feature = self.irreps_r + self.irreps_sh
        self.atomic_embedding = AtomicEmbedding(
            device=device,
            max_z=max_z,
            dim_atoms=dim_atoms,
        )
        self.gaussian = GaussianExpansion(
            device=device,
            mu_min=mu_min,
            mu_max=mu_max,
            step=step,
        )
        self.layers = torch.nn.ModuleList(
            [
                InteractionLayer(
                    device=device,
                    irreps_feature=self.irreps_feature,
                    irreps_atom=irreps_atom,
                    activation_fn=activation_fn,
                )
                for _ in range(n_interactions)
            ]
        )
        self.n_interactions = n_interactions
        self.readout = ReadoutLayer(
            device=device,
            dim_atoms=dim_atoms,
            activation_fn=activation_fn,
        )

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["x"] = tuple(self.atomic_embedding(z) for z in data["z"])
        data["y"] = data["x"]

        graph = data["dgraph"]
        if graph["edges"].numel() > 0:
            graph["ged"] = self.gaussian(graph["r"])
            graph["sh"] = e3nn.o3.spherical_harmonics(
                self.irreps_sh,
                graph["r_hat"],
                normalize=True,
                normalization="component",
            )
            try:
                graph["edge_features"] = torch.cat([graph["ged"], graph["sh"]], dim=-1)
            except:
                raise Exception(f'{graph["ged"].shape=}, {graph["sh"].shape=}')
            data["dgraph"] = graph

            for layer in self.layers:
                data = layer(data)
        readout = self.readout(data)
        return data, readout
