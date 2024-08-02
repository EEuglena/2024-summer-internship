import torch
from torch.utils.data import DataLoader
import e3nn
from torch_geometric.datasets import MD17
from model import InteractionModel
from tqdm import tqdm
import matplotlib.pyplot as plt

DIM_ATOMS = 12
IRREPS_ATOM = e3nn.o3.Irreps(f"{DIM_ATOMS}x0e")
R_CUT = 2.0
MAX_Z = 10
DIM_R = 1
IRREPS_R = e3nn.o3.Irreps(f"{DIM_R}x0e")
IRREPS_SH = e3nn.o3.Irreps.spherical_harmonics(2)
IRREPS_FEATURE = IRREPS_R + IRREPS_SH
DIM_INPUT = 9
DIM_OUTPUT = 30
N_INTERACTIONS = 3

IRREPS_MID = e3nn.o3.Irreps("16x0e + 4x1o + 1x2e")

DATASETS_PATH = "/home/sanghyeonl/git/toy/e3nn_potential/datasets"
CHECKPOINT_PATH = "/home/sanghyeonl/git/toy/e3nn_potential/model.pt"
TARGET_MOLECULE = "revised aspirin"

if __name__ == "__main__":
    raw_data = MD17(DATASETS_PATH, TARGET_MOLECULE)
    model = InteractionModel(
        R_CUT, MAX_Z, DIM_ATOMS, IRREPS_FEATURE, IRREPS_ATOM, IRREPS_MID, N_INTERACTIONS
    )
    train_dataset = [data.to_dict() for data in raw_data[:1000]]
    eval_dataset = [data.to_dict() for data in raw_data[1000:1500]]
    # dataloader = DataLoader(train_dataset, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_loss = []
    best_train_loss = torch.inf
    for epoch in range(5):
        epoch_loss = []
        for i, batch in enumerate(tqdm(train_dataset)):
            data, readout = model(batch)
            loss = (data["energy"] - readout) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        epoch_loss = sum(epoch_loss) / float(len(epoch_loss))
        if epoch_loss < best_train_loss:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                CHECKPOINT_PATH,
            )

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    eval_loss = []
    model.eval()
    for i, batch in enumerate(tqdm(eval_dataset)):
        data, readout = model(batch)
        loss = (data["energy"] - readout) ** 2
        eval_loss.append(loss.item())

    plt.plot(train_loss)
    plt.show()

    print(best_train_loss)
