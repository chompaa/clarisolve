import argparse

import numpy as np
import PIL.Image
import skimage.color
import torch
import torch.backends.cudnn

import models


def colorize(
    model: torch.nn.Module, weights_file: str, output_folder: str, image_file: str
):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    state_dict = model.state_dict()

    for n, p in torch.load(weights_file, map_location=lambda storage, _: storage)[
        "model"
    ].items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval().to(device)

    image_name = image_file.split("/")[-1].split("\\")[-1]

    with PIL.Image.open(image_file).convert("RGB") as image:
        gray_image = skimage.color.rgb2gray(image)

    gray_image = torch.from_numpy(gray_image).to(device).unsqueeze(0).float()
    # gray_image = torch.unsqueeze(gray_image, dim=0)

    # we don't need to calculate gradients during inference (speeds up computation)
    with torch.no_grad():
        predictions = model(gray_image).to(device).clamp(0.0, 1.0).squeeze()

    # expand/reduce predictions dimensionality if it's bigger than the gray image
    if (
        predictions.shape[1] != gray_image.shape[1]
        or predictions.shape[2] != gray_image.shape[2]
    ):
        predictions = torch.nn.functional.interpolate(
            predictions.unsqueeze(0),
            size=(gray_image.shape[1], gray_image.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    output = torch.cat((gray_image, predictions), 0).cpu().numpy()

    output = output.transpose((1, 2, 0))
    output[:, :, 0:1] = output[:, :, 0:1] * 100
    output[:, :, 1:3] = output[:, :, 1:3] * 255 - 128

    output = skimage.color.lab2rgb(output.astype(np.float64))

    PIL.Image.fromarray((output * 255).astype(np.uint8)).save(
        f"{output_folder}{image_name.replace('.', f'_colorized.')}"
    )
    PIL.Image.fromarray(
        (gray_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
    ).save(f"{output_folder}{image_name.replace('.', f'_gray.')}")


if __name__ == "__main__":
    model_list = models.IC_MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=model_list.keys(), required=True)
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    args = parser.parse_args()

    colorize(model_list[args.model](), args.weights_file, "", args.image_file)
