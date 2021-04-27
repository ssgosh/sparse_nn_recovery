import torch
import sys

images = torch.load(sys.argv[1])['images'].to('cpu')
zeros = torch.tensor([-2.429065704345703, -2.418254852294922,
    -2.22139310836792]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
print(zeros, zeros.shape)
print('number of images =', images.shape[0])

n = len(zeros.shape)
per_channel_sparsity = torch.sum(images > zeros, dim=list(range(2, n))).float()
sparsity = torch.sum(images > zeros, dim=list(range(1, n))).float()

#torch.mean(, dim=0)

def print_hist(hist, bins, min, max):
    step = (max - min) / 10
    for i in range(bins):
        print(f'{min + i*step} - {min + (i+1)*step} : {hist[i]}')

def print_sparsity(msg, sparsity):
    mean = torch.mean(sparsity, dim=0).numpy()
    median = torch.median(sparsity, dim=0).values.numpy()
    std = torch.std(sparsity, dim=0).numpy()
    max = torch.max(sparsity, dim=0).values.numpy()
    min = torch.min(sparsity, dim=0).values.numpy()
    hist = torch.histc(sparsity, bins=10, min=0, max=500)
    print(msg, f'mean = {mean}, median = {median}, std = {std}, min = {min}, max = {max}')
    print_hist(hist, 10, 0, 500)

print_sparsity('Total sparisty', sparsity)
print_sparsity('Per-channel sparsity: ', per_channel_sparsity)
