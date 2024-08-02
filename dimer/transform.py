import torch
import numpy as np
import periodictable
from torch_cluster import radius_graph, radius

ELEMENTS_KEY = "elements"
ATOMIC_NUMBER_KEY = "z"
XYZ_KEY = "xyz"
COORDINATES_KEY = "pos"
MONOMER_GRAPH_KEY = "mgraph"
DIMER_GRAPH_KEY = "dgraph"


def symbol_to_atomic_number(symbol):
    element = getattr(periodictable, symbol)
    return element.number


class ExtractAtomicNumber:
    def __init__(
        self, elements_key=ELEMENTS_KEY, atomic_number_key=ATOMIC_NUMBER_KEY
    ) -> None:
        self.elements_key = elements_key
        self.atomic_number_key = atomic_number_key

    def __call__(self, item) -> torch.Any:
        if self.elements_key not in item.keys():
            raise KeyError(self.elements_key)
        elements = item[self.elements_key]
        elements = elements.split()
        z = torch.tensor(
            [symbol_to_atomic_number(element) for element in elements],
            dtype=torch.int64,
        )
        natoms0 = item["natoms0"]
        natoms1 = item["natoms1"]
        z = z.split([natoms0, natoms1], dim=0)
        new_item = item.copy()
        new_item[self.atomic_number_key] = z
        return new_item


class ExtractCoordinates:
    def __init__(self, xyz_key=XYZ_KEY, coordinates_key=COORDINATES_KEY) -> None:
        self.xyz_key = xyz_key
        self.coordinates_key = coordinates_key

    def __call__(self, item) -> torch.Any:
        if self.xyz_key not in item.keys():
            raise KeyError(self.xyz_key)
        if self.coordinates_key in item.keys():
            raise KeyError(self.coordinates_key)
        xyz = item[self.xyz_key]
        xyz = xyz.split()
        xyz = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        xyz = torch.tensor(xyz, dtype=torch.get_default_dtype())
        natoms0 = item["natoms0"]
        natoms1 = item["natoms1"]
        pos = xyz.split([natoms0, natoms1], dim=0)
        new_item = item.copy()
        new_item[self.coordinates_key] = pos
        return new_item


def normalize_edges(edges):
    r = edges.norm(dim=-1)
    r_hat = (
        torch.stack([edge / edge.norm() for edge in edges])
        if edges.numel() > 0
        else torch.empty(0)
    )
    return {"r": r, "r_hat": r_hat}


class MonomerGraph:
    def __init__(
        self,
        coordinates_key=COORDINATES_KEY,
        monomer_graph_key=MONOMER_GRAPH_KEY,
        r_cut=None,
        loop=False,
        n_neighbors=32,
    ) -> None:
        self.coordinates_key = coordinates_key
        self.monomer_graph_key = monomer_graph_key
        self.r_cut = r_cut
        self.loop = loop
        self.n_neighbors = n_neighbors

    def __call__(self, item) -> torch.Any:
        if self.coordinates_key not in item.keys():
            raise KeyError(self.coordinates_key)
        pos = item[self.coordinates_key]
        graphs = []
        for p in pos:
            n_neighbors = self.n_neighbors
            if n_neighbors == 0:
                n_neighbors = p.shape[0]
            src, dst = radius_graph(
                p, r=self.r_cut, loop=self.loop, max_num_neighbors=n_neighbors
            )
            edges = p[dst] - p[src]
            graph = {"src": src, "dst": dst, "edges": edges}
            graph.update(normalize_edges(edges))
            graphs.append(graph)
        new_item = item.copy()
        new_item[self.monomer_graph_key] = graphs
        return new_item


class DimerGraph:
    def __init__(
        self,
        coordinates_key=COORDINATES_KEY,
        dimer_graph_key=DIMER_GRAPH_KEY,
        r_cut=None,
        n_neighbors=32,
    ) -> None:
        self.coordinates_key = coordinates_key
        self.dimer_graph_key = dimer_graph_key
        self.r_cut = r_cut
        self.n_neighbors = n_neighbors

    def __call__(self, item) -> torch.Any:
        if self.coordinates_key not in item.keys():
            raise KeyError(self.coordinates_key)
        pos = item[COORDINATES_KEY]
        n_neighbors = self.n_neighbors
        if n_neighbors == 0:
            n_neighbors = pos[0].shape[0]
        edge_index = radius(
            pos[0],
            pos[1],
            self.r_cut,
            max_num_neighbors=self.n_neighbors + 1,
        )
        src, dst = edge_index[1], edge_index[0]
        edges = pos[0][src] - pos[1][dst]
        graph = {"src": src, "dst": dst, "edges": edges}
        graph.update(normalize_edges(edges))
        new_item = item.copy()
        new_item[self.dimer_graph_key] = graph

        return new_item
