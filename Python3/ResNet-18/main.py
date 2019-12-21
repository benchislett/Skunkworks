import torch
import torchvision
import os

from model import ResNet18
from test import test
from train import train

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

toTensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 2048

dataset_train = torchvision.datasets.CIFAR10(
    root='~/.ml_data', train=True, download=True, transform=toTensor)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = torchvision.datasets.CIFAR10(
    root='~/.ml_data', train=False, download=True, transform=toTensor)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True)

model = ResNet18(10).cuda()

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.99, weight_decay=0.0001)

for epoch in range(32):
    loss_train = train(model, loss, optimizer, train_loader)
    print("Epoch {}  \tLoss: {}".format(epoch, loss_train))

print("Training complete!")

loss_test, acc_test = test(model, loss, test_loader)

print("Test loss: {}\t Test accuracy: {}%".format(loss_test, acc_test))

torch.save(model.state_dict(), os.path.expanduser(
    "~/.ml_data/ResNet-18-STL.pt"))

print("Model saved!")
