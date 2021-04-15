if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

import unittest
from unittest import TestCase

import torch

from utils.torchutils import clip_tensor_range


class Test(TestCase):
    def test_clip_tensor_range(self):
        x = torch.tensor([[0., 1, 2, 3, 4, 5], [0., 1, 2, 3, 4, 5]])
        lower = torch.tensor([2., 1.]).unsqueeze(1)
        upper = torch.tensor([5., 4.]).unsqueeze(1)
        y = clip_tensor_range(x, lower, upper, out=x)
        print(x)
        print(y)


if __name__ == '__main__':
    unittest.main()
