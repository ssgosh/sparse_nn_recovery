if __name__ == '__main__':
    import sys
    sys.path.append(".")

import unittest

import torch

from utils.torchutils import compute_probs_tensor


class TorchUtilsTest(unittest.TestCase):
    def test_compute_probs(self):
        outputs = torch.tensor([
            [0, 1, 2.],
            [3, 4, 5],
            [2, 7, 8],
            [9, 10, 11]
        ])
        targets = torch.tensor([1, 0, 2, 0])
        print((a := compute_probs_tensor(outputs, targets))[0].detach().numpy(), a[1].detach().numpy())


if __name__ == '__main__':
    unittest.main()
