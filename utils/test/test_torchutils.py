if __name__ == '__main__':
    import sys
    sys.path.append(".")

import unittest

import torch

from utils.torchutils import compute_probs_tensor


class TorchUtilsTest(unittest.TestCase):
    def test_compute_probs(self):
        outputs = torch.tensor([
            [0, 1, 0.],
            [1, 1, 0],
            [2, 0, 3]
        ])
        targets = torch.tensor([1, 0, 2])
        print(compute_probs_tensor(outputs, targets).detach().numpy())


if __name__ == '__main__':
    unittest.main()
