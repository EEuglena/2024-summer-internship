from typing import Any
import torch
import pandas as pd
from torch.utils.data import Dataset


class DimerDataset(Dataset):
    def __init__(self, csv_file, dtype, transform=None) -> None:
        super().__init__()
        self.dataframe = pd.read_csv(csv_file, dtype=dtype)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index) -> Any:
        if isinstance(index, int):
            row = self.dataframe.iloc[index]
            row = row.to_dict()
            print(row)
            if self.transform is not None:
                row = self.transform(row)
            return row
        elif isinstance(index, slice):
            rows = self.dataframe.iloc[index]
            rows = rows.to_dict(orient="records")
            if self.transform is not None:
                rows = [self.transform(row) for row in rows]
            return rows
        else:
            raise IndexError("Invalid index type.")
