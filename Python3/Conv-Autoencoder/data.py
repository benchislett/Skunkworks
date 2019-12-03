import torch
import torchvision
import torchvision.datasets as datasets


def get_celeba_loaders(root='~/.ml_data', batch_size=64):
    """Return a tuple containing the attribute names as a tuple,
    and the train, test, and validation data loaders for the
    CelebA dataset from torchvision.
    """

    toTensor = torchvision.transforms.ToTensor()
    train_dset = datasets.CelebA(
        root, split='train', download=True, transform=toTensor)
    test_dset = datasets.CelebA(
        root, split='test', download=True, transform=toTensor)
    val_dset = datasets.CelebA(
        root, split='valid', download=True, transform=toTensor)

    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=batch_size,
                                              shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dset,
                                             batch_size=batch_size,
                                             shuffle=False)
    return (tuple(train_dset.attr_names), train_loader, test_loader, val_loader)
