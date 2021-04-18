import torch
from torch.nn import functional as F

from icontract import require, ensure
from collections import OrderedDict


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


def get_cross(n, c, like):
    """Returns a c x n x n cross X image, on the same device as 'like'"""
    d1 = torch.diagflat(torch.ones(n, device=like.device))
    d2 = torch.flip(d1, dims=[1, ])
    # cross = ((d1 + d2) > 0.).float().unsqueeze(dim=0)
    cross = ((d1 + d2) > 0.).float()
    return torch.stack(c * [cross])


def safe_clone(x):
    return torch.clone(x.detach())


class ClippedConstantTransform(torch.nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        # print(x)
        return torch.clamp(x + self.val, min=0., max=1.)


def load_data_parallel_state_dict_as_normal(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def clip_tensor_range(images, batched_image_zero, batched_image_one, out):
    a = torch.min(images, batched_image_one, out=out)
    b = torch.max(a, batched_image_zero, out=out)
    return b
