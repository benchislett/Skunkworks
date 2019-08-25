import torch
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, Linear, ReLU, Sequential

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.activate = ReLU(inplace=True)

        self.conv1 = Conv2d(in_channels, out_channels,
                            kernel_size=3, stride=stride)
        self.bn1 = BatchNorm2d(out_channels)
        
        self.conv2 = Conv2d(out_channels, out_channels,
                            kernel_size=3, stride=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.shortcut = Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=stride)
        self.shortcut_norm = BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut_norm(self.shortcut(x))

        out = self.activate(out)

        return out

class ResLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, num_blocks=3):
        super(ResLayer, self).__init__()

        blocks = [ResBlock(in_channels, out_channels, stride)]
        for i in range(1, num_blocks - 1):
            layers.append(ResBlock(out_channels, out_channels, stride))
    def forward(self, x):
        for block in blocks:
            x = block(x)
        return x

class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.activate = ReLU(inplace=True)

        self.conv1 = Conv2d(3, 3, kernel_size=7,
                            stride=2, padding=3)
        self.bn1 = BatchNorm2d(3)

        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResLayer(3, 64, stride=1, num_blocks=3)
        self.layer2 = ResLayer(64, 128, stride=2, num_blocks=3)
        self.layer3 = ResLayer(128, 256, stride=2, num_blocks=3)
        self.layer4 = ResLayer(256, 512, stride=2, num_blocks=3)

        self.avgpool = AdaptiveAvgPool2d((1, 1))

        self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x

