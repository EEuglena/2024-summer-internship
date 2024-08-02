from typing import Callable, Dict, Union, Any
import torch
import numpy as np
import e3nn
from torch_scatter import scatter
from torch_cluster import radius_graph
from conversion import kcal2meV


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
        self.tp = e3nn.o3.FullyConnectedTensorProduct(
            irreps_feature, irreps_atom, irreps_atom
        ).to(device)
        self.activation_fn = activation_fn

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = self.tp(data["edge_features"], data["y"][data["dst"]])
        out = out / (data["N"] ** self.exp)
        out = self.activation_fn(out)
        out = scatter(out, data["src"], dim=0, out=torch.zeros_like(data["y"]))
        data["y"] = data["y"] + out
        return data


class ReadoutLayer(torch.nn.Module):
    """Readout layer for interpreting atomic representations into molecular potential energy"""

    def __init__(
        self,
        device: Union[str, torch.device],
        dim_atoms: int,
        dim_mid: int,
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
        self.layer1 = torch.nn.Linear(dim_atoms, dim_mid).to(device)
        self.layer2 = torch.nn.Linear(dim_mid, 1).to(device)
        self.activation_fn = activation_fn

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = self.layer1(data["y"])
        out = self.activation_fn(out)
        out = self.layer2(out)
        return out.sum()


class InteractionModel(torch.nn.Module):
    """Neural Network Model that predicts molecular energy from atomic numbers and coordinates"""

    def __init__(
        self,
        device: Union[str, torch.device],
        activation_fn: Callable,
        r_cut: float,
        max_z: int,
        dim_atoms: int,
        dim_mid: int,
        mu_min: float,
        mu_max: float,
        step: float,
        irreps_r: Union[str, e3nn.o3.Irreps],
        irreps_sh: Union[str, e3nn.o3.Irreps],
        irreps_atom: Union[str, e3nn.o3.Irreps],
        n_interactions: int,
        energy_conversion: Callable = kcal2meV,
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
        self.sign = -1.0
        self.activation_fn = activation_fn
        self.r_cut = r_cut
        self.irreps_r = irreps_r
        self.irreps_sh = irreps_sh
        self.irreps_feature = irreps_r + irreps_sh
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
            dim_mid=dim_mid,
            activation_fn=activation_fn,
        )
        self.energy_conversion = energy_conversion

    def _preprocess(
        self, data: Dict[str, Any], energy_conversion: Callable = kcal2meV
    ) -> Dict[str, Any]:
        N = data["z"].shape[-1]
        src, dst = radius_graph(
            data["pos"],
            self.r_cut,
            loop=True,
            max_num_neighbors=N,
        )
        edges = data["pos"][src] - data["pos"][dst]
        r = edges.norm(dim=-1)
        r = self.gaussian(r)
        sh = e3nn.o3.spherical_harmonics(
            self.irreps_sh, edges, normalize=True, normalization="component"
        )
        edge_features = torch.cat((r, sh), dim=-1)
        data.update(
            {
                "src": src,
                "dst": dst,
                "r": r,
                "sh": sh,
                "edge_features": edge_features,
                "e": energy_conversion(data["energy"]) * self.sign,
                "N": N,
            }
        )
        return data

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = self._preprocess(data, self.energy_conversion)

        data["x"] = self.atomic_embedding(data["z"])

        data["y"] = data["x"]
        for layer in self.layers:
            data = layer(data)
        readout = self.readout(data)
        return data, readout
