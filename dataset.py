#!/usr/bin/env python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ERPDataset(Dataset):
    def __init__(self, dataset_path='./GIB-UVA_ERP-BCI.hdf5', dataset_type="train", val_split=0.2, device="cpu"):
        with h5py.File(dataset_path, 'r') as hf:
            self.features = np.array(hf.get("features"), dtype="float32")
            self.erp_labels = np.array(hf.get("erp_labels"), dtype="long")

        index = int(len(self.erp_labels)*(1.0-val_split))
        if(dataset_type == "train"):
            self.erp_labels = self.erp_labels[:index]
            self.features = self.features[:index]
        elif(dataset_type == "val"):
            self.erp_labels = self.erp_labels[index:]
            self.features = self.features[index:]
        self.erp_labels = torch.from_numpy(self.erp_labels).to(device)
        self.features = torch.from_numpy(self.features.reshape(
            (self.features.shape[0], 1, self.features.shape[1],
             self.features.shape[2])
        )).to(device)

    def __getitem__(self, index):
        return self.features[index], self.erp_labels[index]

    def __len__(self):
        return len(self.erp_labels)
