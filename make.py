import argparse
import glob

import h5py
import numpy as np
import PIL.Image
import skimage.color
import torch

from utils import convert_rgb_to_y


def scale_image(image: PIL.Image.Image, factor: float) -> PIL.Image.Image:
    image = image.resize(
        (int(image.width * factor), int(image.height * factor)),
        resample=PIL.Image.BICUBIC,
    )

    return image


def make_lab_dataset(
    images_dir: str, downscale: bool
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    gray_images = []
    ab_channels = []

    for image_path in sorted(glob.glob(f"{images_dir}/*")):
        with PIL.Image.open(image_path).convert("RGB") as image:
            print(image_path)

            if downscale:
                image = scale_image(image, downscale)

            image = np.array(image)

            gray_image = skimage.color.rgb2gray(image)
            gray_image = torch.from_numpy(gray_image).unsqueeze(0).float()
            gray_images.append(gray_image)

            lab_image = skimage.color.rgb2lab(image)
            # normalize lab image values to [0, 1]
            lab_image = (lab_image + 128) / 255
            channels = lab_image[:, :, 1:]
            channels = torch.from_numpy(channels.transpose(2, 0, 1)).float()
            ab_channels.append(channels)

    return gray_images, ab_channels


def make_ic_train_dataset(
    images_dir: str, output_file: str, patch_size: int, stride: int, downscale: bool
):
    h5_file = h5py.File(output_file, "w")

    gray_patches = []
    ab_patches = []

    for image, ab in zip(*make_lab_dataset(images_dir, downscale)):
        for i in range(0, image.shape[1] - patch_size + 1, stride):
            for j in range(0, image.shape[2] - patch_size + 1, stride):
                print(i, j)
                gray_patches.append(image[:, i : i + patch_size, j : j + patch_size])
                ab_patches.append(ab[:, i : i + patch_size, j : j + patch_size])

    print(gray_patches[0].shape, ab_patches[0].shape)

    gray_patches = np.array(gray_patches)
    ab_patches = np.array(ab_patches)

    h5_file.create_dataset("inputs", data=gray_patches)
    h5_file.create_dataset("labels", data=ab_patches)

    h5_file.close()


def make_ic_eval_dataset(images_dir: str, output_file: str, downscale: bool):
    h5_file = h5py.File(output_file, "w")

    gray_group = h5_file.create_group("inputs")
    ab_group = h5_file.create_group("labels")

    for index, (gray_image, ab_channels) in enumerate(
        zip(*make_lab_dataset(images_dir, downscale))
    ):
        gray_group.create_dataset(str(index), data=gray_image)
        ab_group.create_dataset(str(index), data=ab_channels)

    h5_file.close()


def make_hr_lr_images(
    image_paths: str, scale: int, downscale: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    hr_images = []
    lr_images = []

    for image_path in sorted(glob.glob(f"{image_paths}/*")):
        with PIL.Image.open(image_path).convert("RGB") as hr:
            # downscale images if we want "faster" training
            if downscale:
                hr = scale_image(hr, downscale)

            # want hr image to be divisible by scale
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=PIL.Image.BICUBIC)

            lr = hr.resize(
                (hr_width // scale, hr_height // scale),
                resample=PIL.Image.BICUBIC,
            )
            lr = lr.resize(
                (lr.width * scale, lr.height * scale), resample=PIL.Image.BICUBIC
            )

            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)

            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            hr_images.append(hr)
            lr_images.append(lr)

    return hr_images, lr_images


def make_train_dataset(
    images_dir: str,
    output_file: str,
    patch_size: int,
    stride: int,
    scale: int,
    downscale: float,
):
    h5_file = h5py.File(output_file, "w")

    lr_patches = []
    hr_patches = []

    for hr, lr in zip(*make_hr_lr_images(images_dir, scale, downscale)):
        for i in range(0, lr.shape[0] - patch_size + 1, stride):
            for j in range(0, lr.shape[1] - patch_size + 1, stride):
                lr_patches.append(lr[i : i + patch_size, j : j + patch_size])
                hr_patches.append(hr[i : i + patch_size, j : j + patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("inputs", data=lr_patches)
    h5_file.create_dataset("labels", data=hr_patches)

    h5_file.close()


def make_eval_dataset(images_dir, output_file, scale, downscale):
    h5_file = h5py.File(output_file, "w")

    lr_group = h5_file.create_group("inputs")
    hr_group = h5_file.create_group("labels")

    for index, (hr, lr) in enumerate(
        zip(*make_hr_lr_images(images_dir, scale, downscale))
    ):
        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)

    h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="sr or ic")
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=33)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--downscale", type=float, default=None)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    if args.mode == "sr":
        if args.eval:
            make_eval_dataset(
                args.images_dir, args.output_file, args.scale, args.downscale
            )
        else:
            make_train_dataset(
                args.images_dir,
                args.output_file,
                args.patch_size,
                args.stride,
                args.scale,
                args.downscale,
            )
    elif args.mode == "ic":
        if args.eval:
            make_ic_eval_dataset(args.images_dir, args.output_file, args.downscale)
        else:
            make_ic_train_dataset(
                args.images_dir,
                args.output_file,
                args.patch_size,
                args.stride,
                args.downscale,
            )
    else:
        raise ValueError("mode must be either sr or ic")
