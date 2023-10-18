from torch import Tensor, nn


class ICCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(ICCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        # using L*a*b color space, so 2 channels are output (for a and b)
        self.conv6 = nn.Conv2d(16, 2, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: Tensor) -> Tensor:
        image = self.relu(self.conv1(image))
        image = self.relu(self.conv2(image))
        image = self.relu(self.conv3(image))
        image = self.relu(self.conv4(image))
        image = self.relu(self.conv5(image))
        image = self.conv6(image)

        return image


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: Tensor) -> Tensor:
        image = self.relu(self.conv1(image))
        image = self.relu(self.conv2(image))
        image = self.conv3(image)

        return image
