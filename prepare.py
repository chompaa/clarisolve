import argparse
import glob

import h5py
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import InterpolationMode

from utils import convert_rgb_to_y


def make_hr_lr_images(image_paths, scale, downscale):
    hr_images = []
    lr_images = []

    for image_path in sorted(glob.glob(f"{image_paths}/*")):
        with Image.open(image_path).convert("RGB") as hr:
            # downscale images if we want "faster" training
            if downscale:
                hr = hr.resize(
                    (int(hr.width * downscale), int(hr.height * downscale)),
                    resample=Image.BICUBIC,
                )

            # want hr image to be divisible by scale
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)

            lr = hr.resize(
                (hr_width // scale, hr_height // scale),
                resample=Image.BICUBIC,
            )
            lr = lr.resize(
                (lr.width * scale, lr.height * scale), resample=Image.BICUBIC
            )

            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)

            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            hr_images.append(hr)
            lr_images.append(lr)

    return hr_images, lr_images


def downscale(image, scale):
    transform = transforms.Compose(
        [
            transforms.Resize(
                size=(
                    int(image.height * scale),
                    int(image.width * scale),
                ),
                interpolation=InterpolationMode.BICUBIC,
            ),
        ]
    )

    return


def make_train_dataset(images_dir, output_dir, patch_size, stride, scale, downscale):
    h5_file = h5py.File(output_dir, "w")

    lr_patches = []
    hr_patches = []

    for hr, lr in zip(*make_hr_lr_images(images_dir, scale, downscale)):
        for i in range(0, lr.shape[0] - patch_size + 1, stride):
            for j in range(0, lr.shape[1] - patch_size + 1, stride):
                lr_patches.append(lr[i : i + patch_size, j : j + patch_size])
                hr_patches.append(hr[i : i + patch_size, j : j + patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def make_eval_dataset(images_dir, output_dir, scale, downscale):
    h5_file = h5py.File(output_dir, "w")

    lr_group = h5_file.create_group("lr")
    hr_group = h5_file.create_group("hr")

    for index, (hr, lr) in enumerate(
        zip(*make_hr_lr_images(images_dir, scale, downscale))
    ):
        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)

    h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=33)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--downscale", type=float, default=None)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    if not args.eval:
        make_train_dataset(
            args.images_dir,
            args.output_dir,
            args.patch_size,
            args.stride,
            args.scale,
            args.downscale,
        )
    else:
        make_eval_dataset(args.images_dir, args.output_dir, args.scale, args.downscale)
