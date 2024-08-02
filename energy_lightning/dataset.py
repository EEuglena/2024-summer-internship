import torch
from torch_geometric.datasets import MD17
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl


class MD17DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        name,
        batch_size,
        train_size,
        val_size,
        test_size,
        pred_size,
        num_workers,
    ):
        super().__init__()
        self.KCAL_TO_MCAL = 0.001
        self.root = root
        self.name = name
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.pred_size = pred_size
        self.num_workers = num_workers
        # self.std = None
        # self.mu = None

    def prepare_data(self):
        raw_dataset = MD17(root=self.root, name=self.name)
        # train_dataset = raw_dataset[: self.train_size]
        # self.std, self.mu = torch.std_mean(
        #     torch.tensor([data.energy for data in train_dataset], dtype=float)
        # )

    def setup(self, stage):
        raw_dataset = MD17(root=self.root, name=self.name)
        dataset = self._pre_proc(raw_dataset)
        self.train_ds, self.val_ds, self.test_ds, self.pred_ds = random_split(
            dataset,
            [
                self.train_size,
                self.val_size,
                self.test_size,
                self.pred_size,
            ],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _pre_proc(self, raw_dataset):
        dataset = raw_dataset[
            : self.train_size + self.val_size + self.test_size + self.pred_size
        ]
        dataset = [data.to_dict() for data in dataset]
        for data in dataset:
            data["energy"] *= self.KCAL_TO_MCAL
            data["r"] = self._pos_to_dist(data["pos"])
        return dataset

    def _pos_to_dist(self, pos):
        pos1 = pos.unsqueeze(0)
        pos2 = pos.unsqueeze(1)

        rpos = pos1 - pos2
        distance = torch.sum(rpos.square(), 2).sqrt()

        return distance

    # def get_ref_value(self):
    #     return {"std": self.std, "mu": self.mu}
