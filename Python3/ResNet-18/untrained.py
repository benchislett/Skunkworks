import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU

class ResBlock(nn.Module):

    def __init__(self, in_depth, out_depth, stride=1):
        super(ResBlock, self).__init__()

        self.activate = ReLU(inplace = True)

        self.conv1 = Conv2d()
        self.bn1 = BatchNorm2d()
        
        self.conv2 = Conv2d()
        self.bn2 = BatchNorm2d()

        self.shortcut = Conv2d()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.activate(out)

        return out


