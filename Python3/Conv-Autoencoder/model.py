import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE(nn.Module):
    def __init__(self, code_bits=1024, width=32, height=32):
        super(CAE, self).__init__()

        self.w = width
        self.h = height
        self.imgsize = width * height * 3
        self.code_bits = code_bits

        # Encoder
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(24, 16, 3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(8, 3, 3, padding=1, stride=1)
        self.linear_in = nn.Linear(self.imgsize // 4, code_bits)

        # Decoder
        self.linear_out = nn.Linear(code_bits, self.imgsize // 4)
        self.deconv1 = nn.ConvTranspose2d(3, 8, 3, padding=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, padding=1, stride=1)
        self.deconv3 = nn.ConvTranspose2d(16, 24, 3, padding=1, stride=1)
        self.deconv4 = nn.ConvTranspose2d(24, 16, 3, padding=1, stride=1)
        self.deconv5 = nn.ConvTranspose2d(16, 8, 4, padding=1, stride=2)
        self.deconv6 = nn.ConvTranspose2d(8, 3, 3, padding=1, stride=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, self.imgsize // 4)
        x = F.sigmoid(self.linear_in(x))
        return x

    def decode(self, x):
        x = F.relu(self.linear_out(x)).view(-1, 3, self.w // 2, self.h // 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = F.sigmoid(self.deconv6(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
