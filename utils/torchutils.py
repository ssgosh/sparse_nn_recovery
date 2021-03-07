import torch
from torch.nn import functional as F


def compute_probs_tensor(output, targets):
    #two_d_indices = torch.cat([torch.arange(targets.shape[0], device=targets.device).unsqueeze(1), targets.unsqueeze(1)], dim=1)
    #print(two_d_indices)
    #chunked = two_d_indices.chunk(chunks=3, dim=0)
    #print(chunked)
    softmax = F.softmax(output, dim=1)
    print(targets.shape)
    print(output.shape)
    print(softmax.shape)
    # For gather(), index must have the same number of dimensions as the input.
    # We'll just repeat target num_classes times along dimension 1
    targets = torch.cat([targets.unsqueeze(1)]*output.shape[1], dim=1)
    print(targets)
    return output.gather(1, targets)
