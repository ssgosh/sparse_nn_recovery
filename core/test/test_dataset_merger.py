if __name__ == "__main__":
    import sys
    sys.path.append(".")

import unittest
from unittest import TestCase
import pytest

import torch
from torch.utils.data import TensorDataset, DataLoader

from core.adversarial_dataset_manager import DatasetMerger


class TestDatasetMerger(TestCase):
    def test_dataset_merger(self):
        dmerger = DatasetMerger(0.7, True)
        for i in range(10):
            x = torch.randn(10, 1, 2, 2)
            t = torch.randint(low=0, high=10, size=[10])
            new_train = DataLoader(TensorDataset(x, t), batch_size=x.shape[0])
            combined = dmerger.combine_with_previous_train(new_train)
            #print(combined.batch_size, new_train.batch_size)
            assert combined.batch_size == new_train.batch_size
            N = len(combined.dataset)
            n = len(new_train.dataset)
            if i > 0:
                assert n != N
            else:
                assert n == N

            a_new, b_new = next(iter(new_train))
            a_comb, b_comb = next(iter(DataLoader(combined.dataset, batch_size=len(combined.dataset))))
            assert check_subset(a_new, a_comb)
            assert check_subset(b_new, b_comb)
            #print("\n".join([str(row) for row in a_new]))
            #print("\n".join([str(row) for row in a_comb]))
            #print("\n".join([str(row) for row in b_new]))
            #print("\n".join([str(row) for row in b_comb]))



def check_subset(x, y):
    """
    Check if tensor x is a subset of y

    XXX: works only if
    :param x:
    :param y:
    :return:
    """
    p = True
    for a in x:
        present = False
        for b in y:
            present = torch.all(a == b).item()
            if present: break
        assert present, "{a}, {b}"
        p = p and present
    return p


if __name__ == "__main__":
    #pytest.main(['-c', 'pytest.ini', 'core/test/test_dataset_merger.py'])
    unittest.main()


