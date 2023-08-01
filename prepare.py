import argparse
import glob

import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils import convert_rgb_to_y


def get_image_paths(path):
    return sorted(glob.glob(f"{path}/*"))


def make_hr_lr_images(image_paths, scale):
    hr_images = []
    lr_images = []

    # floating point scale factors are bad..
    scale = int(scale)

    for image_path in get_image_paths(image_paths):
        with Image.open(image_path).convert("RGB") as hr:
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


def train(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_patches = []
    hr_patches = []

    for hr, lr in zip(*make_hr_lr_images(args.images_dir, args.scale)):
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i : i + args.patch_size, j : j + args.patch_size])
                hr_patches.append(hr[i : i + args.patch_size, j : j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_group = h5_file.create_group("lr")
    hr_group = h5_file.create_group("hr")

    for index, (hr, lr) in enumerate(
        zip(*make_hr_lr_images(args.images_dir, args.scale))
    ):
        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)

    h5_file.close()


def resize(args):
    for image_path in get_image_paths(args.images_dir):
        with Image.open(image_path) as image:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=(
                            int(image.height * args.scale),
                            int(image.width * args.scale),
                        ),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.ToTensor(),
                ]
            )

            image = transform(image)
            name = image_path.split("/")[-1].split(".")[0].split("\\")[-1]

            save_image(image, f"{args.output_path}/{name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=33)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--scale", type=float, default=2)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()

    if not args.eval and not args.resize:
        train(args)
    elif args.resize:
        resize(args)
    else:
        eval(args)
