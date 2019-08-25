import torch

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1):
    out = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    torch.nn.init.kaiming_normal_(out.weight, mode='fan_out',
                                  nonlinearity='relu')
    return out

def BatchNorm2d(size):
    out = torch.nn.BatchNorm2d(size)

    torch.nn.init.constant_(out.weight, 1)
    torch.nn.init.constant_(out.bias, 0)

    return out

class ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.activate = torch.nn.ReLU(inplace=True)

        self.conv1 = Conv2d(in_channels, out_channels,
                            kernel_size=3, stride=stride)
        self.bn1 = BatchNorm2d(out_channels)
        
        self.conv2 = Conv2d(out_channels, out_channels,
                            kernel_size=3, stride=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.shortcut = Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=stride, padding=0)
        self.shortcut_norm = BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut = self.shortcut_norm(self.shortcut(x))

        out += shortcut

        out = self.activate(out)

        return out

class ResLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, num_blocks=3):
        super(ResLayer, self).__init__()

        self.blocks = [ResBlock(in_channels, out_channels, stride)]
        for i in range(1, num_blocks - 1):
            self.blocks.append(ResBlock(out_channels, out_channels, stride))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResNet18(torch.nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.activate = torch.nn.ReLU(inplace=True)

        self.conv1 = Conv2d(3, 3, kernel_size=7,
                            stride=2, padding=3)
        self.bn1 = BatchNorm2d(3)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResLayer(3, 64, stride=1, num_blocks=3)
        self.layer2 = ResLayer(64, 128, stride=2, num_blocks=3)
        self.layer3 = ResLayer(128, 256, stride=2, num_blocks=3)
        self.layer4 = ResLayer(256, 512, stride=2, num_blocks=3)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = torch.nn.Linear(512, num_classes)

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

