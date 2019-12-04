def test_batch(model, loss_fn, batch):
    """Test the model on a batch of data,
    and return the loss and number of correct predictions

    """
    x = batch[0].cuda()

    out = model(x)

    loss = loss_fn(out, x).item()

    return loss


def test(model, loss_fn, loader):
    """Test the model over all batches in a given dataset,
    and return the total mean loss and prediction accuracy

    """
    model.eval()

    loss_acc = 0.0
    batches = 0
    for batch in loader:
        loss = test_batch(model, loss_fn, batch)

        loss_acc += loss
        batches += 1

    return loss_acc / batches
