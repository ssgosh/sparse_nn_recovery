import torch
from torch.nn import functional as F

from icontract import require, ensure


# Input contract
@require(lambda output: len(output.shape) == 2)
@require(lambda targets: len(targets.shape) == 1)
@require(lambda output, targets: targets.shape[0] == output.shape[0])
# Output contract
@ensure(lambda result, targets: result[0].shape[0] == targets.shape[0])
@ensure(lambda result, targets: result[1].shape[0] == targets.shape[0])
def compute_probs_tensor(output, targets):
    softmax = F.softmax(output, dim=1)
    # For gather(), index must have the same number of dimensions as the input.
    # We'll just repeat target num_classes times along dimension 1
    targets = torch.cat([targets.unsqueeze(1)] * output.shape[1], dim=1)
    return output.gather(1, targets)[:, 0], softmax.gather(1, targets)[:, 0]


def get_cross(n, like):
    """Returns an n x n cross X image, on the same device as 'like'"""
    d1 = torch.diagflat(torch.ones(n, device=like.device))
    d2 = torch.flip(d1, dims=[1, ])
    cross = ((d1 + d2) > 0.).float().unsqueeze(dim=0)
    return cross