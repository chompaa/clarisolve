import torch


def calculate_psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        x (torch.Tensor): Input image 1.
        y (torch.Tensor): Input image 2.

    Returns:
        psnr (torch.Tensor): Peak Signal-to-Noise Ratio.
    """

    if (x.shape != y.shape) or (len(x.shape) != 4):
        # pad tensors to match shapes
        x = torch.nn.functional.pad(
            x, (0, y.shape[3] - x.shape[3], 0, y.shape[2] - x.shape[2])
        )
        y = torch.nn.functional.pad(
            y, (0, x.shape[3] - y.shape[3], 0, x.shape[2] - y.shape[2])
        )

    return 10.0 * torch.log10(1.0 / torch.mean((x - y) ** 2))


class AverageMeter(object):
    # taken from the PyTorch ImageNet example

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
