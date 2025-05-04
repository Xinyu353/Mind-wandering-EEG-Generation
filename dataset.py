# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None).values
        if pd.isnull(self.data).any():
            raise ValueError("Data contains NaN values. Please clean the data.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

if __name__ == "__main__":
    dataset = MyDataset(csv_file='processed_file.csv')
    print(f"Dataset loaded with {len(dataset)} samples")
