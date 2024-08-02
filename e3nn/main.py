import e3nn
import torch
from torch_cluster import radius_graph
from e3nn import o3
from torch_geometric.datasets import MD17
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from pprint import pprint

R_CUT = 2.0
SH_IRREPS = o3.Irreps.spherical_harmonics(2)


class InteractionLayer(torch.nn.Module):
    def __init__(
        self,
        n_interaction,
    ):
        super().__init__()
        self.n_interaction = n_interaction
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(12, 1) for _ in range(n_interaction)]
        )

    def forward(
        self,
        batch,
    ):
        batch["x"] = batch["z"].copy()
        for layer in self.layers:
            for z, src, dst, r, r_hat in zip(
                batch["z"], batch["src"], batch["dst"], batch["r"], batch["r_hat"]
            ):
                z_src = torch.tensor([z[s] for s in src])
                z_dst = torch.tensor([z[d] for d in dst])
                inpt = torch.cat([z_src, z_dst, r.flatten(), r_hat.flatten()], dim)


class ReadoutLayer(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.layer = torch.nn.Linear(12, 1)

    def forward(
        self,
        batch,
    ):
        readout = self.layer(batch["x"])

        return readout


class EquivariantModule(torch.nn.Module):
    def __init__(
        self,
        r_cut,
    ):
        super().__init__()
        self.r_cut = r_cut
        self.readout = ReadoutLayer()

    def _preprocess(
        self,
        batch,
    ):

        graphs = [radius_graph(pos, self.r_cut).tolist() for pos in batch["pos"]]
        src = [graph[0] for graph in graphs]
        dst = [graph[1] for graph in graphs]
        batch["src"] = src
        batch["dst"] = dst

        edges = []
        for pos, s, d in zip(batch["pos"], src, dst):
            edge = torch.tensor([(pos[j] - pos[i]).tolist() for i, j in zip(s, d)])
            edges.append(edge)
        r = [e.norm(dim=-1) for e in edges]
        r_hat = o3.spherical_harmonics(
            SH_IRREPS, edge, normalize=True, normalization="component"
        )
        batch["r"] = r
        batch["r_hat"] = r_hat

        return batch

    def forward(
        self,
        batch,
    ):

        batch = self._preprocess(batch)
        readout = self.readout(batch)

        return readout


if __name__ == "__main__":
    model = EquivariantModule(R_CUT)

    raw_data = MD17("/home/sanghyeonl/git/toy/e3nn/datasets/", "revised aspirin")
    dataset = [data.to_dict() for data in raw_data[:100]]
    dataloader = DataLoader(dataset, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    for i, batch in enumerate(tqdm(dataloader)):
        output = model(batch)
        loss = criterion(output, torch.tensor([data["energy"] for data in batch]))

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Batch {i + 1} : {loss=:.3f}")
