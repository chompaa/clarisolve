import torch


class _ResBlock(torch.nn.Module):
    def __init__(self):
        super(_ResBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2)
        self.in1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.in2 = torch.nn.BatchNorm2d(64)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        identity = image

        image = self.relu(self.in1(self.conv1(image)))
        image = self.in2(self.conv2(image))

        image = torch.add(image, identity)

        return image


class SRRes(torch.nn.Module):
    def __init__(self):
        super(SRRes, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.residual = self.make_layer(_ResBlock, 4, 1)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = torch.nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def make_layer(self, block, num_blocks, num_channels):
        layers = [block() for _ in range(num_blocks)]

        return torch.nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        identity = image

        image = self.relu(self.conv1(image))
        image = self.residual(image)
        image = self.relu(self.conv2(image))
        image = self.conv3(image)
        image = torch.add(image, identity)

        return image

    def __str__(self):
        return "srres"
