if __name__ == '__main__':
    import sys
    sys.path.append(".")

import unittest

import torch
import torch.nn.functional as F

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

        outs, probs = a
        for i in range(outputs.shape[0]):
            soft = F.softmax(outputs[i], dim=0)
            assert probs[i] == soft[targets[i]]
            assert outs[i] == outputs[i, targets[i]]

if __name__ == '__main__':
    unittest.main()
