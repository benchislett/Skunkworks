import torch
import torch.nn.functional as F


def test_batch(model, loss_fn, batch):
    """Test the model on a batch of data,
    and return the loss and number of correct predictions

    """
    x, y = batch[0].cuda(), batch[1].cuda()

    out = model(x)
    # out = F.softmax(out, dim=-1)
    _, pred = torch.max(out, 1)

    loss = loss_fn(out, y).item()
    count = torch.sum(pred == y)

    return loss, count


def test(model, loss_fn, loader):
    """Test the model over all batches in a given dataset,
    and return the total mean loss and prediction accuracy

    """
    model.eval()

    loss_acc = 0.0
    count_acc = 0.0
    batches = 0
    for batch in loader:
        loss, count = test_batch(model, loss_fn, batch)

        loss_acc += loss
        count_acc += count
        batches += 1

    return loss_acc / batches, count_acc / batches
