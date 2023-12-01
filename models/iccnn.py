import torch


class ICCNN(torch.nn.Module):
    def __init__(self):
        super(ICCNN, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        )

        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(32, 2, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, output):
        output = self.features(output)
        output = self.upsample(output)

        return output

    def __str__(self):
        return "iccnn"
