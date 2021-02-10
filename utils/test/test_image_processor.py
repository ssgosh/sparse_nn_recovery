import sys
sys.path.append(".")

from utils import image_processor as imp

import torch

def test_get_sparsity_batch():
    x = torch.zeros(3, 1, 2, 2)
    x[0][0][0][0] = 5.
    x[0][0][1][0] = 3.
    x[1][0][0][1] = -3.5
    x[1][0][1][1] = .5
    zero = 0.
    ret = imp.get_sparsity_batch(x, zero)
    assert torch.all(ret == torch.tensor([2, 1, 0]))

test_get_sparsity_batch()
