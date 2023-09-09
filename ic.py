import argparse

import numpy as np
import PIL.Image
import skimage.color
import torch
import torch.backends.cudnn

import models


def ic(weights_file: str, output_folder: str, image_file: str):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.ICCNN().to(device)

    state_dict = model.state_dict()

    for n, p in torch.load(
        weights_file, map_location=lambda storage, _: storage
    ).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image_name = image_file.split("/")[-1].split("\\")[-1]

    with PIL.Image.open(image_file).convert("RGB") as image:
        gray_image = skimage.color.rgb2gray(image)

    gray_image = torch.from_numpy(gray_image).unsqueeze(0).float()
    # gray_image = torch.unsqueeze(gray_image, dim=0)

    # we don't need to calculate gradients during inference (speeds up computation)
    with torch.no_grad():
        predictions = model(gray_image).clamp(0.0, 1.0)

    output = torch.cat((gray_image, predictions), 0).numpy()

    output = output.transpose((1, 2, 0))
    output[:, :, 0:1] = output[:, :, 0:1] * 100
    output[:, :, 1:3] = output[:, :, 1:3] * 255 - 128

    output = skimage.color.lab2rgb(output.astype(np.float64))

    PIL.Image.fromarray((output * 255).astype(np.uint8)).save(
        f"{output_folder}{image_name.replace('.', f'_colorized.')}"
    )
    PIL.Image.fromarray((gray_image.squeeze().numpy() * 255).astype(np.uint8)).save(
        f"{output_folder}{image_name.replace('.', f'_gray.')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    args = parser.parse_args()

    ic(args.weights_file, "", args.image_file)
