import torch
import torchvision
import os

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from data import get_celeba_loaders
from model import CAE
from test import test
from train import train

batch_size = 128
num_epochs = 8

attr_names, train_loader, test_loader, val_loader = get_celeba_loaders(
    batch_size=batch_size)

model = CAE(code_bits=1024, width=218, height=178)

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.99, weight_decay=0.0001)

for epoch in range(num_epochs):
    epoch_loss = train(model, loss, optimizer, train_loader)
    print("[Epoch {}] Loss: {}".format(epoch, epoch_loss))
