import torch, torchvision

from model import ResNet18

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

toTensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensor)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=toTensor)
test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=True)

model = ResNet18(10).cuda()

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):
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

        print("Epoch: {}\tBatch: {}, Batch Loss: {}".format(epoch, i, loss_tmp))

    print("\n\nEpoch {} Complete! Loss: {}\n\n".format(epoch, loss_acc / num_batches))

print("Training complete!")

