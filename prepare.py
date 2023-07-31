import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from pathlib import Path
from utils import convert_rgb_to_y


def train(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob(f"{args.images_dir}/*")):
        hr = pil_image.open(image_path).convert("RGB")
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_group = h5_file.create_group("lr")
    hr_group = h5_file.create_group("hr")

    for i, image_path in enumerate(sorted(glob.glob(f"{args.images_dir}/*"))):
        hr = pil_image.open(image_path).convert("RGB")
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


def resize(args):
  image_path_list = list(glob.glob(f"{args.images_dir}/*.png"))

  for image_path in image_path_list:
    with Image.open(image_path) as image:
      transform = transforms.Compose([
          transforms.Resize(size=(int(image.height * args.resize_scale), int(image.width * args.resize_scale)), interpolation=Image.BICUBIC),
          transforms.ToTensor(),
      ])

      image = transform(image)
      name = image_path.split('/')[-1].split('.')[0].split("\\")[-1]

      save_image(image, f"{args.output_path}/{name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=33)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--resize-scale", type=float, default=0.1)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if not args.eval and not args.resize:
        train(args)
    elif args.resize:
        resize(args)
    else:
        eval(args)