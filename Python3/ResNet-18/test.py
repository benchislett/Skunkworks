import torch, torchvision
import torch.nn.functional as F
import numpy as np
import os

from model import ResNet18

torch.cuda.set_device(0)
torch.set_default_tensor_type("torch.cuda.FloatTensor")

model = ResNet18(10).cuda()
model.load_state_dict(torch.load(os.path.expanduser("~/.ml_data/ResNet-18-STL.pt")))

model.eval()

toTensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size=100

dataset_test = torchvision.datasets.STL10(root="~/.ml_data", split="test", download=True, transform=toTensor)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

acc_total = 0
acc_count = 0

for i, batch in enumerate(test_loader):
    x, y = batch[0].cuda(), batch[1].cuda()

    out = F.softmax(model(x), dim=-1)

    _, pred = torch.max(out, 1)

    acc_total += torch.sum(pred == y)
    acc_count += batch_size

print("Total accuracy: {}%".format(100 * acc_total / acc_count))

