import numpy as np
import torch
import torchvision


class ICRes(torch.nn.Module):
    def __init__(self, blocks=2):
        super(ICRes, self).__init__()

        assert blocks >= 0 and blocks <= 4

        midlevel_feature_size = 64 * blocks

        # first half: ResNet-18
        resnet = torchvision.models.resnet18(num_classes=365)
        # modify first the initial convolution layer to take a single-channel input
        resnet.conv1.weight = torch.nn.Parameter(
            resnet.conv1.weight.sum(dim=1).unsqueeze(1)
        )
        # include initial convolution and pooling layers
        num_layers = 2 + (blocks * 2)
        # extract the first six layers from ResNet-18 (up to )
        self.midlevel_resnet = torch.nn.Sequential(
            *list(resnet.children())[0:num_layers]
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

        # second half: upsampling
        # self.upsample = torch.nn.Sequential(
        #     torch.nn.Conv2d(
        #         midlevel_feature_size, 128, kernel_size=3, stride=1, padding=1
        #     ),
        #     torch.nn.BatchNorm2d(128),
        #     torch.nn.ReLU(),
        #     torch.nn.Upsample(scale_factor=2),
        #     torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     torch.nn.BatchNorm2d(64),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     torch.nn.BatchNorm2d(64),
        #     torch.nn.ReLU(),
        #     torch.nn.Upsample(scale_factor=2),
        #     torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     torch.nn.BatchNorm2d(32),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
        #     torch.nn.Upsample(scale_factor=2),
        # )

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
