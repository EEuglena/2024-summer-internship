from typing import Any, Dict, List
import os
import numpy as np
import random
import torch
import periodictable

from torch_geometric.data import Dataset


def InteractionDataLoader(dataset: Dataset, name: str = None) -> List[Dict[str, Any]]:
    if hasattr(dataset, "name"):
        name = dataset.name
    if name == None:
        raise Exception("Name of the molecule must be given.")
    return [{**data.to_dict(), "name": name} for data in dataset]


def parseXYZ(device, file_name):
    with open(file_name, "r") as file:
        data = file.readlines()
    U0 = float(data[1].split()[12].replace("*^", "e"))
    pos = []
    z = []
    for d in data[2:-3]:
        v = d.split()
        element = v[0]
        _x, _y, _z = [float(i.replace("*^", "e")) for i in v[1:-1]]
        pos.append(torch.tensor([_x, _y, _z], dtype=torch.get_default_dtype()))
        z.append(periodictable.elements.symbol(element).number)
    pos = torch.stack(pos).to(device)
    z = torch.tensor(z).to(device)
    energy = torch.tensor(U0, dtype=torch.get_default_dtype()).to(device)
    return {"pos": pos, "z": z, "energy": energy}


def GDB9Dataset(root):
    file_names = []
    dir_names = os.listdir(root)
    for dir_name in dir_names:
        if dir_name.endswith(".xyz"):
            file_names.append(os.path.join(root, dir_name))
    return file_names


def GDB9DataLoader(device, file_names, train_size, eval_size, test_size):
    idx = random.sample(
        np.arange(len(file_names)).tolist(), train_size + eval_size + test_size
    )
    idx = np.split(
        idx, [train_size, train_size + eval_size, train_size + eval_size + test_size]
    )
    train_idx = idx[0]
    eval_idx = idx[1]
    test_idx = idx[2]
    np.random.shuffle(train_idx)
    train_dataset = [parseXYZ(device, file_names[i]) for i in train_idx]
    eval_dataset = [parseXYZ(device, file_names[i]) for i in eval_idx]
    test_dataset = [parseXYZ(device, file_names[i]) for i in test_idx]
    return train_dataset, eval_dataset, test_dataset
