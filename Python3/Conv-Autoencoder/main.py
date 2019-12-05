from logger import Logger
from train import train
from test import test
from model import CAE
from data import get_celeba_loaders
import torch
import torchvision
import os

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


batch_size = 128
num_epochs = 32

attr_names, train_loader, test_loader, val_loader = get_celeba_loaders(
    batch_size=batch_size)
train_logger = Logger(len(train_loader))

model = CAE(code_bits=2048, width=218, height=178).to('cuda:0')

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.00001)

for epoch in range(num_epochs):
    epoch_loss = train(model, loss, optimizer, train_loader, train_logger)
    train_logger.epoch(epoch, epoch_loss)

torch.save(model.state_dict(), os.path.expanduser("~/.ml_data/Conv-Autoencoder-CelebA.pt"))

print("Model Saved.")
