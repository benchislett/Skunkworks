def train_batch(model, loss_fn, optimizer, batch):
    """Train the model on a single batch of data,
    and return the total loss

    """
    optimizer.zero_grad()

    x = batch[0].cuda()

    pred = model(x)
    loss_batch = loss_fn(pred, x)
    loss_batch.backward()
    optimizer.step()

    return loss_batch.item()


def train(model, loss_fn, optimizer, loader, logger):
    """Train the model over all batches in a given dataset,
    and return the total loss

    """
    model.train()

    loss_acc = 0.0
    batches = 0
    for idx, batch in enumerate(loader):
        loss = train_batch(model, loss_fn, optimizer, batch)

        logger.batch(idx, loss)

        loss_acc += loss
        batches += 1

    return loss_acc / batches
