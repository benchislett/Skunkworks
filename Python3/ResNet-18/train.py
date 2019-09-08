import torch, torchvision
import os

from model import ResNet18

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

toTensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_train = torchvision.datasets.STL10(root='~/.ml_data', split="train", download=True, transform=toTensor)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_test = torchvision.datasets.STL10(root='~/.ml_data', split="test", download=True, transform=toTensor)

model = ResNet18(10).cuda()

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(64):
    loss_acc = 0.0
    num_batches = 0
    for i, batch in enumerate(train_loader):
        x, y = batch[0].cuda(), batch[1].cuda()

        optimizer.zero_grad()

        pred = model(x)
        loss_batch = loss(pred, y)
        loss_batch.backward()
        optimizer.step()

        loss_tmp = loss_batch.item()
        loss_acc += loss_tmp
        num_batches += 1

    print("Epoch {}\t Loss: {}".format(epoch, loss_acc / num_batches))

print("Training complete!")

torch.save(model.state_dict(), os.path.expanduser("~/.ml_data/ResNet-18-STL.pt"))

print("Model saved!")
