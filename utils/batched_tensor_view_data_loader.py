import torch


# Creates and supplies random chunks of a tensor
# Gradient flows to the original tensor 
# PyTorch doesn't support this because it does batching with torch.stack()
class BatchedTensorViewDataLoader:

    # data is a list of tensors of size N, which needs to be batched
    def __init__(self, batch_size, *data):
        self.batch_size = batch_size
        self.data = data
        self.N = data[0].shape[0]
        assert self.batch_size <= self.N
        for dataitem in data:
            assert self.N == dataitem.shape[0]

        self.perm_indices = torch.arange(self.N)
        self.cur_idx = 0


    # Get next batch
    # Leaves last small batch
    def next(self):
        end = self.cur_idx + self.batch_size
        next_idx = end
        if end > self.N:
            self.cur_idx = 0
            end = self.batch_size
            self.perm_indices = torch.randperm(self.N)
        idx = self.perm_indices[self.cur_idx:end]
        self.cur_idx = end
        return [item[idx] for item in self.data]


if __name__ == '__main__':
    x = torch.arange(4, dtype=torch.float32, requires_grad=True)
    y = torch.arange(4)
    z = torch.arange(4)
    loader = BatchedTensorViewDataLoader(3, x, y, z)
    for i in range(10):
        a, b, c = loader.next()
        print(a, b, c)


