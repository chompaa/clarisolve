import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, index):
        with h5py.File(self.h5_file, "r") as dataset_file:
            return np.expand_dims(dataset_file["lr"][index] / 255.0, 0), np.expand_dims(
                dataset_file["hr"][index] / 255.0, 0
            )

    def __len__(self):
        with h5py.File(self.h5_file, "r") as dataset_file:
            return len(dataset_file["lr"])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, index):
        with h5py.File(self.h5_file, "r") as dataset_file:
            return np.expand_dims(
                dataset_file["lr"][str(index)][:, :] / 255.0, 0
            ), np.expand_dims(dataset_file["hr"][str(index)][:, :] / 255.0, 0)

    def __len__(self):
        with h5py.File(self.h5_file, "r") as dataset_file:
            return len(dataset_file["lr"])
