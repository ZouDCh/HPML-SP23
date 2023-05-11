import torch
from torch.utils.data import Dataset
import os
import time
import sys
import json

# A custom dataset for reading the wikifact dataset
class WikiFactDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        self.inputs = []
        self.targets = []

        self.read_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx]),
            "targets": torch.tensor(self.targets[idx]),
        }

    def read_data(self):
        self.inputs = json.load(open(self.data_path+"_ids.txt"))
        self.targets = json.load(open(self.data_path+"_labels.txt"))

