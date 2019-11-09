import torch
import torchvision.datasets as datasets


def get_celeba_loaders(root='~/.ml_data', batch_size=64, num_workers=2):
    """Return a tuple containing the attribute names as a tuple,
    and the train, test, and validation data loaders for the
    CelebA dataset from torchvision.
    """

    train_dset = datasets.CelebA(root, split='train', download=True)
    test_dset = datasets.CelebA(root, split='test', download=True)
    val_dset = datasets.CelebA(root, split='valid', download=True)

    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    return (tuple(train_dset.attr_names), train_loader, test_loader, val_loader)
