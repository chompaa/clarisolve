import numpy as np
import torch
import torchvision


class ICRes(torch.nn.Module):
    def __init__(self, blocks=2):
        super(ICRes, self).__init__()

        assert blocks >= 0 and blocks <= 4

        midlevel_feature_size = 64 * np.power(2, blocks - 1)

        # first half: ResNet-18
        resnet = torchvision.models.resnet18(num_classes=365)
        # modify first the initial convolution layer to take a single-channel input
        resnet.conv1.weight = torch.nn.Parameter(
            resnet.conv1.weight.sum(dim=1).unsqueeze(1)
        )
        # include initial convolution and pooling layers
        num_layers = blocks * 2
        if blocks == 2:
            num_layers += 2
        # extract the first n layers from ResNet-18
        self.midlevel_resnet = torch.nn.Sequential(
            *list(resnet.children())[:num_layers]
        )

        layers = [
            torch.nn.Conv2d(
                midlevel_feature_size, midlevel_feature_size, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(midlevel_feature_size),
        ]

        for i in range(blocks):
            in_channels = int(midlevel_feature_size / np.power(2, i))
            out_channels = int(midlevel_feature_size / np.power(2, i + 1))
            print(in_channels, out_channels)

            layers.extend(
                (
                    torch.nn.Upsample(scale_factor=2),
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                )
            )

        layers.extend(
            (
                torch.nn.Conv2d(
                    int(midlevel_feature_size / np.power(2, blocks)),
                    2,
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.Upsample(scale_factor=2),
            )
        )

        self.upsample = torch.nn.Sequential(*layers)

    def forward(self, output):
        # add dim to input if wrong dimension
        if output.dim() < 4:
            output = output.unsqueeze(0)

        # pass input through our ResNet to extract features
        midlevel_features = self.midlevel_resnet(output)

        # upsample to get colors
        output = self.upsample(midlevel_features)

        return output

    def __str__(self):
        return "icres"
