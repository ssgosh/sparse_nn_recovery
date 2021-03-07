import torch
from torch.nn import functional as F


def compute_probs_tensor(output, targets):
    two_d_indices = torch.cat([torch.arange(targets.shape[0], device=targets.device).unsqueeze(1), targets.unsqueeze(1)], dim=1)
    print(two_d_indices)
    #chunked = two_d_indices.chunk(chunks=3, dim=0)
    #print(chunked)
    return F.softmax(output, dim=1).gather(two_d_indices)
