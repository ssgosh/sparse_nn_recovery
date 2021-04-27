import torch
import sys

images = torch.load(sys.argv[1])['images'].to('cpu')
zeros = torch.tensor([-2.429065704345703, -2.418254852294922,
    -2.22139310836792]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
print(zeros, zeros.shape)

n = len(zeros.shape)
per_channel_sparsity = torch.mean(torch.sum(images > zeros, dim=list(range(2,
    n))).float(), dim=0)
sparsity = torch.mean(torch.sum(images > zeros, dim=list(range(1,
    n))).float(), dim=0)

print('Total sparisty', sparsity)
print('Per-channel sparsity: ', per_channel_sparsity)
