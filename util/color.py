import numpy as np
import torch


def convert_rgb_to_y(img):
    # y channel calculation taken from https://en.wikipedia.org/wiki/YCbCr

    if type(img) == np.ndarray:
        return (
            16.0
            + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2])
            / 256.0
        )
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return (
            16.0
            + (64.738 * img[0, ...] + 129.057 * img[1, ...] + 25.064 * img[2, ...])
            / 256.0
        )
    else:
        raise Exception("Unknown Type", type(img))


def convert_rgb_to_ycbcr(img):
    # calculations taken from https://en.wikipedia.org/wiki/YCbCr

    if type(img) == np.ndarray:
        y = (
            16.0
            + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2])
            / 256.0
        )
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = (
            16.0
            + (64.738 * img[0, ...] + 129.057 * img[1, ...] + 25.064 * img[2, ...])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[0, ...] - 74.494 * img[1, ...] + 112.439 * img[2, ...])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[0, ...] - 94.154 * img[1, ...] - 18.285 * img[2, ...])
            / 256.0
        )
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(img))


def convert_ycbcr_to_rgb(img):
    # inverse operation of calculation from https://en.wikipedia.org/wiki/YCbCr

    if type(img) == np.ndarray:
        r = 298.082 * img[..., 0] / 256.0 + 408.583 * img[..., 2] / 256.0 - 222.921
        g = (
            298.082 * img[..., 0] / 256.0
            - 100.291 * img[..., 1] / 256.0
            - 208.120 * img[..., 2] / 256.0
            + 135.576
        )
        b = 298.082 * img[..., 0] / 256.0 + 516.412 * img[..., 1] / 256.0 - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, ...] / 256.0 + 408.583 * img[2, ...] / 256.0 - 222.921
        g = (
            298.082 * img[0, ...] / 256.0
            - 100.291 * img[1, ...] / 256.0
            - 208.120 * img[2, ...] / 256.0
            + 135.576
        )
        b = 298.082 * img[0, ...] / 256.0 + 516.412 * img[1, ...] / 256.0 - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(img))
