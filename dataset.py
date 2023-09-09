import h5py
import numpy as np
import torch.utils.data


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, input_key, label_key, normalize=True):
        super(TrainDataset, self).__init__()

        self.h5_file = h5_file
        self.inputs_key = input_key
        self.labels_key = label_key
        self.normalize = normalize

    def __getitem__(self, index: int):
        with h5py.File(self.h5_file, "r") as dataset_file:
            inputs = dataset_file[self.inputs_key]
            labels = dataset_file[self.labels_key]

            if not isinstance(inputs, h5py.Dataset) or not isinstance(
                labels, h5py.Dataset
            ):
                raise TypeError("Train dataset is not a h5py.Dataset")

            if self.normalize:
                return np.expand_dims(inputs[index] / 255.0, 0), np.expand_dims(
                    labels[index] / 255.0, 0
                )
            else:
                return (inputs[index], labels[index])

    def __len__(self):
        with h5py.File(self.h5_file, "r") as dataset_file:
            inputs = dataset_file[self.inputs_key]

            if not isinstance(inputs, h5py.Dataset):
                return 0

            return len(inputs)


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, input_key, label_key, normalize=True):
        super(EvalDataset, self).__init__()

        self.h5_file = h5_file
        self.inputs_key = input_key
        self.labels_key = label_key
        self.normalize = normalize

    def __getitem__(self, index: int):
        with h5py.File(self.h5_file, "r") as dataset_file:
            inputs = dataset_file[self.inputs_key]
            labels = dataset_file[self.labels_key]

            if not isinstance(inputs, h5py.Group) or not isinstance(labels, h5py.Group):
                raise TypeError("Eval dataset does not contain h5py.Group items")

            input_dataset = inputs[str(index)]
            label_dataset = labels[str(index)]

            if not isinstance(input_dataset, h5py.Dataset) or not isinstance(
                label_dataset, h5py.Dataset
            ):
                raise TypeError("Eval dataset is not a h5py.Dataset")

            if self.normalize:
                return np.expand_dims(input_dataset[:, :] / 255.0, 0), np.expand_dims(
                    label_dataset[:, :] / 255.0, 0
                )
            else:
                return (input_dataset[:, :], label_dataset[:, :])

    def __len__(self):
        with h5py.File(self.h5_file, "r") as dataset_file:
            inputs = dataset_file[self.inputs_key]

            if not isinstance(inputs, h5py.Group):
                return 0

            return len(inputs)
