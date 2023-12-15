import argparse

import numpy as np
import PIL.Image
import torch
import torch.backends.cudnn

import models
import util


def super_resolve(
    model: torch.nn.Module,
    weights_file: str,
    output_folder: str,
    image_file: str,
    scale: int,
    upscale: bool,
):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

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
        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale
        image = image.resize((image_width, image_height), resample=PIL.Image.BICUBIC)

        if not upscale:
            image = image.resize(
                (image.width // scale, image.height // scale),
                resample=PIL.Image.BICUBIC,
            )

        image = image.resize(
            (image.width * scale, image.height * scale),
            resample=PIL.Image.BICUBIC,
        )

        image.save(f"{output_folder}{image_name.replace('.', f'_bicubic_x{scale}.')}")

        image = np.array(image).astype(np.float32)

        ycbcr = util.convert_rgb_to_ycbcr(image)

    y_channel = ycbcr[..., 0] / 255.0
    y_channel = torch.from_numpy(y_channel).to(device)
    y_channel = y_channel.unsqueeze(0).unsqueeze(0)

    # we don't need to calculate gradients during inference (speeds up computation)
    with torch.no_grad():
        predictions = model(y_channel).clamp(0.0, 1.0)

    psnr = None

    # only calculate PSNR if we have ground truth
    if not upscale:
        psnr = util.calculate_psnr(y_channel, predictions)

    predictions = predictions.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([predictions, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(util.convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

    PIL.Image.fromarray(output).save(
        f"{output_folder}{image_name.replace('.', f'_{model}_x{scale}.')}"
    )

    return psnr


if __name__ == "__main__":
    model_list = models.SR_MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=model_list.keys(), required=True)
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--upscale", action="store_true", default=True)
    args = parser.parse_args()

    super_resolve(
        model_list[args.model](),
        args.weights_file,
        args.output_dir,
        args.image_file,
        args.scale,
        args.upscale,
    )
