import h5py
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets
import skimage.color
import PIL.ImageFile


class ColorDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        path, target = self.imgs[index]
        img = self.loader(path)

        img_light = None
        img_ab = None

        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

        if self.transform is not None:
            img = self.transform(img)
            img = np.array(img)

            img_light = skimage.color.rgb2gray(img)
            img_light = torch.from_numpy(img_light).unsqueeze(0).float()

            img_lab = skimage.color.rgb2lab(img)
            img_lab = (img_lab + 128) / 255

            img_ab = img_lab[:, :, 1:]
            img_ab = torch.from_numpy(img_ab.transpose(2, 0, 1)).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_light, img_ab


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, input_key, label_key, normalize=True):
        super(TrainDataset, self).__init__()

        self.h5_file = h5_file
        self.inputs_key = input_key
        self.labels_key = label_key
        self.normalize = normalize

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
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

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
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
