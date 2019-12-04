class Logger():
    def __init__(self, num_batches):
        self.num_batches = num_batches

    def batch(self, batch_idx, loss):
        fmt = "[Batch {} of {}] Loss: {}"
        print(fmt.format(batch_idx + 1, self.num_batches, loss))

    def epoch(self, epoch_idx, loss):
        fmt = "\n[Epoch {}] Loss: {}\n"
        print(fmt.format(epoch_idx + 1, loss))
