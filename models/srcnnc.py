import torch


class SRCNNC(torch.nn.Module):
    def __init__(self):
        super(SRCNNC, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = torch.nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, output):
        identity = output

        output = self.relu(self.conv1(output))
        output = self.relu(self.conv2(output))
        output = self.conv3(output)

        output = torch.add(output, identity)
        output = self.relu(output)

        return output

    def __str__(self):
        return "srcnnc"
